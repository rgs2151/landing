import { ArrowUpRight } from 'lucide-react';
import { profile, paper } from '../content/site';
import ProjectTitle from './ProjectTitle';

function ToolFigure({ figure, showMedia = true, inlineFigures = [] }) {
  const Animation = figure.Animation;
  return <div className="pb-tool-body" data-figure={figure.id}>
    {showMedia && <div className="pb-tool-media"><Animation /></div>}
    <div className={`pb-blurb${inlineFigures.length ? ' pb-illustrated-abstract' : ''}`}>
      {inlineFigures.map(({ id, Animation, align, afterLines }) => <div key={id}
        className="pb-abstract-figure" data-side={align} data-figure={id}
        style={{ '--figure-offset': `${afterLines}lh` }}><Animation /></div>)}
      {figure.text.split(/\n\s*\n/).map((text, index) => {
      const citationIndex = figure.citation ? text.indexOf(figure.citation.text) : -1;
      return <p key={index}>{citationIndex < 0 ? text : <>
        {text.slice(0, citationIndex)}
        <a href={figure.citation.url} target="_blank" rel="noopener noreferrer">{figure.citation.text}</a>
        {text.slice(citationIndex + figure.citation.text.length)}
      </>}</p>;
    })}</div>
  </div>;
}

export default function ToolDetail({ tool, headingRef }) {
  const Gallery = tool.Gallery;
  return <article className="pb-project pb-tool" aria-labelledby="pb-project-heading">
    <h2 id="pb-project-heading" ref={headingRef} tabIndex={-1}><ProjectTitle project={tool} /></h2>
    <p className="pb-authors">{tool.authors.map((author, index) => <span key={author}>
      {index > 0 ? ', ' : ''}<span className={author === profile.name ? 'pb-self' : undefined}>{author}</span>
    </span>)}</p>
    <div className="pb-project-links">
      <a href={tool.code} target="_blank" rel="noopener noreferrer"><img src={paper.githubLogo} alt="GitHub" /> code<ArrowUpRight size={17} /></a>
      {tool.resourceLinks?.map(link => <a key={link.url} href={link.url} target="_blank" rel="noopener noreferrer">{link.label}<ArrowUpRight size={17} /></a>)}
      <a href={tool.url} target="_blank" rel="noopener noreferrer">Documentation<ArrowUpRight size={17} /></a>
    </div>
    {Gallery && <div className="pb-tool-gallery"><Gallery /></div>}
    {tool.abstract ? <ToolFigure figure={{ id: tool.id, text: tool.abstract }} showMedia={false} inlineFigures={tool.abstractFigures} />
      : tool.figures.map(figure => <ToolFigure key={figure.id} figure={figure} showMedia={!Gallery} />)}
  </article>;
}
