import { githubSVG, emailSVG, linkedInSVG, discordSVG } from "@/app/svg";

export default function Footer() {
  return (
    <footer>
      <span className="font-semibold text-lg">contact</span>
      <div>
        {/* <!-- Contact --> */}
        <div className="contact">
          <a href="https://github.com/a-r-r-o-w" target="_blank" rel="noreferrer" aria-label="GitHub">{githubSVG}</a>
          <a href="mailto:contact.aryanvs@gmail.com" target="_blank" rel="noreferrer" aria-label="Email">{emailSVG}</a>
          <a href="https://linkedin.com/in/aryan-v-s" target="_blank" rel="noreferrer" aria-label="LinkedIn">{linkedInSVG}</a>
          <a href="https://discordapp.com/users/761868631021584406" target="_blank" rel="noreferrer" aria-label="Discord">{discordSVG}</a>
        </div>
      </div>
    </footer>
  );
}
