import About from "../../components/about";

export default function AboutRoute() {
  return (
    // <!-- Main Content -->
    <main>
      <About 
        description={`
          Hi! I'm Aryan, a mostly self-taught programmer with good algorithmic, problem solving and development
          experience. I'm passionate about generative AI, reinforcement learning, cybersecurity, performance
          engineering, cryptography, graphics, game development and competitive programming.
        `}
        additional_info="This site is under development, please check back later."
      />
    </main>
  )
}
