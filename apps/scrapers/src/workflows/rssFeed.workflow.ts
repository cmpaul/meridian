import { $articles, $sources, inArray } from '@meridian/database';
import { WorkflowEntrypoint, WorkflowEvent, WorkflowStep, WorkflowStepConfig } from 'cloudflare:workers';
import { err, ok, ResultAsync } from 'neverthrow';
import { Env } from '../index';
import { parseRSSFeed } from '../lib/parsers';
import { getRssFeedWithFetch } from '../lib/puppeteer';
import { DomainRateLimiter } from '../lib/rateLimiter';
import { getDb } from '../lib/utils';
import { startProcessArticleWorkflow } from './processArticles.workflow';

type Params = { force?: boolean };

const tierIntervals = {
  1: 60 * 60 * 1000, // Tier 1: Check every hour
  2: 4 * 60 * 60 * 1000, // Tier 2: Check every 4 hours
  3: 6 * 60 * 60 * 1000, // Tier 3: Check every 6 hours
  4: 24 * 60 * 60 * 1000, // Tier 4: Check every 24 hours
};

const dbStepConfig: WorkflowStepConfig = {
  retries: { limit: 3, delay: '1 second', backoff: 'linear' },
  timeout: '5 seconds',
};

// Takes in a rss feed URL, parses the feed & stores the data in our database.
export class ScrapeRssFeed extends WorkflowEntrypoint<Env, Params> {
  async run(_event: WorkflowEvent<Params>, step: WorkflowStep) {
    try {
      console.log('Starting RSS feed scraping workflow...');
      const db = getDb(this.env.DATABASE_URL);

      // Fetch all sources
      console.log('Fetching sources from database...');
      const feeds = await step.do('get feeds', dbStepConfig, async () => {
        try {
          let feeds = await db
            .select({
              id: $sources.id,
              lastChecked: $sources.lastChecked,
              scrape_frequency: $sources.scrape_frequency,
              url: $sources.url,
            })
            .from($sources);
          console.log(`Found ${feeds.length} total feeds in database`);

          if (_event.payload.force === undefined || _event.payload.force === false) {
            feeds = feeds.filter(feed => {
              if (feed.lastChecked === null) {
                return true;
              }
              const lastCheckedTime =
                feed.lastChecked instanceof Date ? feed.lastChecked.getTime() : new Date(feed.lastChecked).getTime();
              const interval = tierIntervals[feed.scrape_frequency as keyof typeof tierIntervals] || tierIntervals[2];
              return Date.now() - lastCheckedTime >= interval;
            });
            console.log(`${feeds.length} feeds need to be checked based on their scrape frequency`);
          }

          return feeds.map(e => ({ id: e.id, url: e.url }));
        } catch (error) {
          console.error('Error fetching feeds from database:', error);
          throw error;
        }
      });

      if (feeds.length === 0) {
        console.log('All feeds are up to date, exiting early...');
        return;
      }

      // Process feeds with rate limiting
      const now = Date.now();
      const oneWeekAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);
      const allArticles: Array<{ sourceId: number; link: string; pubDate: Date | null; title: string }> = [];

      // Create rate limiter with RSS feed specific settings
      const rateLimiter = new DomainRateLimiter<{ id: number; url: string }>({
        maxConcurrent: 10,
        globalCooldownMs: 500,
        domainCooldownMs: 2000,
      });

      console.log('Starting to process feeds with rate limiting...');
      // Process feeds with rate limiting
      const feedResults = await rateLimiter.processBatch(feeds, step, async (feed, _domain) => {
        try {
          console.log(`Processing feed ${feed.id}: ${feed.url}`);
          return await step.do(
            `scrape feed ${feed.id}`,
            {
              retries: { limit: 3, delay: '2 seconds', backoff: 'exponential' },
            },
            async () => {
              try {
                const feedPage = await getRssFeedWithFetch(feed.url);
                if (feedPage.isErr()) {
                  console.error(`Error fetching feed ${feed.url}:`, feedPage.error);
                  throw feedPage.error;
                }

                const feedArticles = await parseRSSFeed(feedPage.value);
                if (feedArticles.isErr()) {
                  console.error(`Error parsing feed ${feed.url}:`, feedArticles.error);
                  throw feedArticles.error;
                }

                const filteredArticles = feedArticles.value
                  .filter(({ pubDate }) => pubDate === null || pubDate > oneWeekAgo)
                  .map(e => ({ ...e, sourceId: feed.id }));

                console.log(`Found ${filteredArticles.length} new articles from feed ${feed.id}`);
                return filteredArticles;
              } catch (error) {
                console.error(`Error processing feed ${feed.id}:`, error);
                throw error;
              }
            }
          );
        } catch (error) {
          console.error(`Error in rate limiter for feed ${feed.id}:`, error);
          return [];
        }
      });

      // Flatten the results into allArticles
      feedResults.forEach(articles => {
        allArticles.push(...articles);
      });
      console.log(`Total new articles found: ${allArticles.length}`);

      // Insert articles and update sources
      if (allArticles.length > 0) {
        console.log('Inserting new articles into database...');
        await step.do('insert new articles', dbStepConfig, async () => {
          try {
            await db
              .insert($articles)
              .values(
                allArticles.map(({ sourceId, link, pubDate, title }) => ({
                  sourceId,
                  url: link,
                  title,
                  publishDate: pubDate,
                }))
              )
              .onConflictDoNothing();
            console.log('Successfully inserted new articles');
          } catch (error) {
            console.error('Error inserting articles:', error);
            throw error;
          }
        });

        console.log('Updating source lastChecked timestamps...');
        await step.do('update sources', dbStepConfig, async () => {
          try {
            await db
              .update($sources)
              .set({ lastChecked: new Date() })
              .where(inArray($sources.id, Array.from(new Set(allArticles.map(({ sourceId }) => sourceId)))));
            console.log('Successfully updated source timestamps');
          } catch (error) {
            console.error('Error updating source timestamps:', error);
            throw error;
          }
        });
      } else {
        console.log('No new articles found, updating source timestamps...');
        await step.do('update sources with no articles', dbStepConfig, async () => {
          try {
            await db
              .update($sources)
              .set({ lastChecked: new Date() })
              .where(
                inArray(
                  $sources.id,
                  feeds.map(feed => feed.id)
                )
              );
            console.log('Successfully updated source timestamps');
          } catch (error) {
            console.error('Error updating source timestamps:', error);
            throw error;
          }
        });
      }

      console.log('Triggering article processor workflow...');
      await step.do('trigger_article_processor', dbStepConfig, async () => {
        try {
          const workflow = await startProcessArticleWorkflow(this.env);
          if (workflow.isErr()) {
            console.error('Error starting article processor workflow:', workflow.error);
            throw workflow.error;
          }
          console.log('Successfully triggered article processor workflow');
          return workflow.value.id;
        } catch (error) {
          console.error('Error in article processor workflow:', error);
          throw error;
        }
      });

      console.log('RSS feed scraping workflow completed successfully');
    } catch (error) {
      console.error('Fatal error in RSS feed scraping workflow:', error);
      throw error;
    }
  }
}

export async function startRssFeedScraperWorkflow(env: Env, params?: Params) {
  const workflow = await ResultAsync.fromPromise(env.SCRAPE_RSS_FEED.create({ id: crypto.randomUUID(), params }), e =>
    e instanceof Error ? e : new Error(String(e))
  );
  if (workflow.isErr()) {
    console.error('Error creating RSS feed scraper workflow:', workflow.error);
    return err(workflow.error);
  }
  return ok(workflow.value);
}
