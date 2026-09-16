import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
    summary: z.string(),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    type: z.enum(['project', 'paper']).default('project'),
    status: z.string(),
    venue: z.string().optional(),
    year: z.number().optional(),
    authors: z.string().optional(),
    order: z.number().default(0),
    basedOn: z.array(z.string()).optional(),
    links: z
      .object({
        github: z.string().optional(),
        demo: z.string().optional(),
        paper: z.string().optional(),
        arxiv: z.string().optional(),
      })
      .optional(),
    summary: z.string(),
  }),
});

export const collections = { posts, projects };
