import { ArrowUpRight } from 'lucide-react';
import { profile, paper } from '../content/site';

export default function ToolDetail({ tool, headingRef }) {
  const Animation = tool.Animation;
  const citationIndex = tool.citation ? tool.blurb.indexOf(tool.citation.text) : -1;
  return <article className="pb-project pb-tool" aria-labelledby="pb-project-heading">
    <h2 id="pb-project-heading" ref={headingRef} tabIndex={-1}>{tool.title}</h2>
    <p className="pb-authors">{tool.authors.map((author, index) => <span key={author}>
      {index > 0 ? ', ' : ''}<span className={author === profile.name ? 'pb-self' : undefined}>{author}</span>
    </span>)}</p>
    <div className="pb-project-links">
      <a href={tool.url} target="_blank" rel="noopener noreferrer">Documentation<ArrowUpRight size={17} /></a>
      <a href={tool.code} target="_blank" rel="noopener noreferrer"><img src={paper.githubLogo} alt="GitHub" /> code<ArrowUpRight size={17} /></a>
    </div>
    <div className="pb-tool-body">
      <div className="pb-tool-media"><Animation /></div>
      <div className="pb-blurb"><p>{citationIndex < 0 ? tool.blurb : <>
        {tool.blurb.slice(0, citationIndex)}
        <a href={tool.citation.url} target="_blank" rel="noopener noreferrer">{tool.citation.text}</a>
        {tool.blurb.slice(citationIndex + tool.citation.text.length)}
      </>}</p></div>
    </div>
  </article>;
}
