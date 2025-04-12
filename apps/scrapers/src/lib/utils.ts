import { getDb as getDbFromDatabase } from '@meridian/database';
import { Context, MiddlewareHandler } from 'hono';
import { ResultAsync } from 'neverthrow';
import { HonoEnv } from '../app';

export function getDb(databaseUrl: string) {
  return getDbFromDatabase(databaseUrl, { prepare: false });
}

export const safeFetch = ResultAsync.fromThrowable(
  (url: string, returnType: 'text' | 'json' = 'text', options: RequestInit = {}) =>
    fetch(url, options).then(res => {
      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      if (returnType === 'text') return res.text();
      return res.json();
    }),
  e => (e instanceof Error ? e : new Error(String(e)))
);

export const randomBetween = (min: number, max: number) => min + Math.random() * (max - min);

/**
 * Escape special characters for XML
 */
export function escapeXml(unsafe: string): string {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export function cleanString(text: string) {
  return text
    .replace(/[ \t]+/g, ' ') // collapse spaces/tabs
    .replace(/\n\s+/g, '\n') // clean spaces after newlines
    .replace(/\s+\n/g, '\n') // clean spaces before newlines
    .replace(/\n{3,}/g, '\n\n') // keep max 2 consecutive newlines
    .trim(); // clean edges
}

export function cleanUrl(url: string) {
  const u = new URL(url);

  const paramsToRemove = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid', 'gclid'];
  paramsToRemove.forEach(param => u.searchParams.delete(param));

  return u.toString();
}

export function hasValidAuthToken(c: Context<HonoEnv>) {
  const auth = c.req.header('Authorization');
  if (auth === undefined || auth !== `Bearer ${c.env.MERIDIAN_SECRET_KEY}`) {
    return false;
  }
  return true;
}

export const corsMiddleware: MiddlewareHandler<HonoEnv> = async (c, next) => {
  const origin = c.env.CORS_ORIGIN || '*';

  // Handle preflight requests
  if (c.req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Max-Age': '86400',
      },
    });
  }

  // Add CORS headers to all responses
  await next();
  c.res.headers.set('Access-Control-Allow-Origin', origin);
  c.res.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  c.res.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
};
