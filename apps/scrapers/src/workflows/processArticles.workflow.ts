import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { $articles, and, eq, gte, isNull, sql } from '@meridian/database';
import { generateObject } from 'ai';
import { WorkflowEntrypoint, WorkflowEvent, WorkflowStep, WorkflowStepConfig } from 'cloudflare:workers';
import { err, ok, ResultAsync } from 'neverthrow';
import { Env } from '../index';
import { getArticleWithBrowser, getArticleWithFetch } from '../lib/puppeteer';
import { DomainRateLimiter } from '../lib/rateLimiter';
import { disposeDb, getDb } from '../lib/utils';
import getArticleAnalysisPrompt, { articleAnalysisSchema } from '../prompts/articleAnalysis.prompt';

const dbStepConfig: WorkflowStepConfig = {
  retries: { limit: 3, delay: '1 second', backoff: 'linear' },
  timeout: '5 seconds',
};

// Main workflow class
export class ProcessArticles extends WorkflowEntrypoint<Env, Params> {
  async run(_event: WorkflowEvent<Params>, step: WorkflowStep) {
    const env = this.env;
    console.log('Starting article processing workflow...');
    const db = getDb(env.DATABASE_URL);
    const client = db.$client;
    console.log(`Connected to database: ${this.env.DATABASE_URL.split('@')[1]?.split('/')[0] || 'unknown'}`);
    const google = createGoogleGenerativeAI({ apiKey: env.GOOGLE_API_KEY, baseURL: env.GOOGLE_BASE_URL });

    try {
      async function getUnprocessedArticles(opts: { limit?: number }) {
        console.log('Fetching unprocessed articles from database...');
        const query = db
          .select({
            id: $articles.id,
            url: $articles.url,
            title: $articles.title,
            publishedAt: $articles.publishDate,
          })
          .from($articles)
          .where(
            and(
              // only process articles that haven't been processed yet
              isNull($articles.processedAt),
              // only process articles that have a publish date in the last 48 hours
              gte($articles.publishDate, new Date(new Date().getTime() - 48 * 60 * 60 * 1000)),
              // only articles that have not failed
              isNull($articles.failReason)
            )
          )
          .limit(opts.limit ?? 100)
          .orderBy(sql`RANDOM()`);

        console.log(`Executing query: ${query.toSQL().sql}`);
        console.log(`Query parameters: ${JSON.stringify(query.toSQL().params)}`);

        const articles = await query;
        console.log(`Found ${articles.length} unprocessed articles`);
        return articles;
      }

      // get articles to process
      const articles = await step.do('get articles', dbStepConfig, async () => getUnprocessedArticles({ limit: 200 }));

      if (articles.length === 0) {
        console.log('No unprocessed articles found, exiting workflow');
        return;
      }

      // Create rate limiter with article processing specific settings
      console.log('Initializing rate limiter for article processing...');
      const rateLimiter = new DomainRateLimiter<{
        id: number;
        url: string;
        title: string;
        publishedAt: Date | null;
      }>({
        maxConcurrent: 8,
        globalCooldownMs: 1000,
        domainCooldownMs: 5000,
      });

      const articlesToProcess: Array<{
        id: number;
        title: string;
        text: string;
        publishedTime?: string;
      }> = [];

      const trickyDomains = [
        'reuters.com',
        'nytimes.com',
        'politico.com',
        'science.org',
        'alarabiya.net',
        'reason.com',
        'telegraph.co.uk',
        'lawfaremedia',
        'liberation.fr',
        'france24.com',
      ];

      // Process articles with rate limiting
      console.log('Starting to process articles with rate limiting...');
      const articleResults = await rateLimiter.processBatch(articles, step, async (article, domain) => {
        // Skip PDF files immediately
        if (article.url.toLowerCase().endsWith('.pdf')) {
          console.log(`Skipping PDF file: ${article.url}`);
          return { id: article.id, success: false, error: 'pdf' };
        }

        console.log(`Processing article ${article.id} from domain ${domain}`);
        const result = await step.do(
          `scrape article ${article.id}`,
          {
            retries: { limit: 3, delay: '2 second', backoff: 'exponential' },
            timeout: '1 minute',
          },
          async () => {
            // start with light scraping
            let articleData: { title: string; text: string; publishedTime: string | undefined } | undefined = undefined;

            // if we're from a tricky domain, fetch with browser first
            if (trickyDomains.includes(domain)) {
              console.log(`Using browser rendering for tricky domain: ${domain}`);
              const articleResult = await getArticleWithBrowser(env, article.url);
              if (articleResult.isErr()) {
                console.error(`Browser rendering failed for ${article.url}:`, articleResult.error);
                throw articleResult.error;
              }
              articleData = articleResult.value;
              console.log(`Successfully scraped article ${article.id} using browser rendering`);
            } else {
              // otherwise, start with fetch & then browser if that fails
              console.log(`Attempting light scraping for ${article.url}`);
              const lightResult = await getArticleWithFetch(article.url);
              if (lightResult.isErr()) {
                console.log(`Light scraping failed, falling back to browser rendering for ${article.url}`);
                // rand jitter between .5 & 3 seconds
                const jitterTime = Math.random() * 2500 + 500;
                await step.sleep(`jitter`, jitterTime);

                const articleResult = await getArticleWithBrowser(env, article.url);
                if (articleResult.isErr()) {
                  console.error(`Browser rendering failed for ${article.url}:`, articleResult.error);
                  throw articleResult.error;
                }

                articleData = articleResult.value;
                console.log(
                  `Successfully scraped article ${article.id} using browser rendering after light scraping failed`
                );
              } else {
                articleData = lightResult.value;
                console.log(`Successfully scraped article ${article.id} using light scraping`);
              }
            }

            return { id: article.id, success: true, html: articleData };
          }
        );

        return result;
      });

      // Handle results
      console.log('Processing scraping results...');
      for (const result of articleResults) {
        if (result.success && 'html' in result) {
          console.log(`Successfully scraped article ${result.id}, adding to processing queue`);
          articlesToProcess.push({
            id: result.id,
            title: result.html.title,
            text: result.html.text,
            publishedTime: result.html.publishedTime,
          });
        } else {
          console.log(`Failed to scrape article ${result.id}, marking as failed with reason: ${result.error}`);
          // update failed articles in DB with the fail reason
          await step.do(`update db for failed article ${result.id}`, dbStepConfig, async () => {
            await db
              .update($articles)
              .set({
                processedAt: new Date(),
                failReason: result.error ? String(result.error) : 'Unknown error',
              })
              .where(eq($articles.id, result.id));
            console.log(`Updated database for failed article ${result.id}`);
          });
        }
      }

      // process with LLM
      console.log(`Starting LLM processing for ${articlesToProcess.length} articles...`);
      await Promise.all(
        articlesToProcess.map(async article => {
          console.log(`Processing article ${article.id} with LLM...`);
          const articleAnalysis = await step.do(
            `analyze article ${article.id}`,
            {
              retries: { limit: 3, delay: '2 seconds', backoff: 'exponential' },
              timeout: '1 minute',
            },
            async () => {
              console.log(`Generating analysis for article ${article.id}...`);
              const response = await generateObject({
                model: google('gemini-2.0-flash'),
                temperature: 0,
                prompt: getArticleAnalysisPrompt(article.title, article.text),
                schema: articleAnalysisSchema,
              });
              console.log(`Successfully generated analysis for article ${article.id}`);
              return response.object;
            }
          );

          // update db
          console.log(`Updating database for article ${article.id}...`);
          await step.do(`update db for article ${article.id}`, dbStepConfig, async () => {
            await db
              .update($articles)
              .set({
                processedAt: new Date(),
                content: article.text,
                title: article.title,
                completeness: articleAnalysis.completeness,
                relevance: articleAnalysis.relevance,
                language: articleAnalysis.language,
                location: articleAnalysis.location,
                summary: (() => {
                  if (articleAnalysis.summary === undefined) return null;
                  let txt = '';
                  txt += `HEADLINE: ${articleAnalysis.summary.headline.trim()}\n`;
                  txt += `ENTITIES: ${articleAnalysis.summary.entities.join(', ')}\n`;
                  txt += `EVENT: ${articleAnalysis.summary.event.trim()}\n`;
                  txt += `CONTEXT: ${articleAnalysis.summary.context.trim()}\n`;
                  return txt.trim();
                })(),
              })
              .where(eq($articles.id, article.id))
              .execute();
            console.log(`Successfully updated database for article ${article.id}`);
          });
        })
      );

      console.log(`Successfully processed ${articlesToProcess.length} articles with LLM`);

      // check if there are articles to process still
      console.log('Checking for remaining unprocessed articles...');
      const remainingArticles = await step.do('get remaining articles', dbStepConfig, async () =>
        getUnprocessedArticles({ limit: 100 })
      );
      if (remainingArticles.length > 0) {
        console.log(`Found ${remainingArticles.length} remaining articles to process, triggering new workflow...`);

        // trigger the workflow again
        await step.do('trigger_article_processor', dbStepConfig, async () => {
          const workflow = await this.env.PROCESS_ARTICLES.create({ id: crypto.randomUUID() });
          console.log(`Triggered new article processor workflow with ID: ${workflow.id}`);
          return workflow.id;
        });
      } else {
        console.log('No remaining articles to process, workflow complete');
      }
    } finally {
      // Ensure database connection is properly disposed
      await client.end();
      disposeDb();
      console.log('Database connection closed in ProcessArticles');
    }
  }
}

// helper to start the workflow from elsewhere
export async function startProcessArticleWorkflow(env: Env) {
  const workflow = await ResultAsync.fromPromise(env.PROCESS_ARTICLES.create({ id: crypto.randomUUID() }), e =>
    e instanceof Error ? e : new Error(String(e))
  );
  if (workflow.isErr()) {
    return err(workflow.error);
  }
  return ok(workflow.value);
}
