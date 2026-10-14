export default function ProjectTitle({ project, title = project.title, showLogo = true }) {
  return <>{showLogo && project.logo && <img className="pb-project-logo" src={project.logo} alt="" />}{title}</>;
}
