import Image from "next/image";

import missingGradientImage from "./utils";

export default function PaperImplementation({name, index}) {
  let [canPlayVideo, setCanPlayVideo] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    setCanPlayVideo(document.createElement("video").canPlayType("video/mp4") !== "");
  }, []);

  let displayContent = missingGradientImage(name, index);

  if (image !== null)
    displayContent = <Image className="cards__img" src={image} alt={name} />;
  if (video !== null && canPlayVideo) {
    displayContent = (
      <video
        className="cards__vid"
        ref={videoRef}
        loop muted playsInline
        alt={name}
        src={video}
      >
        Your browser does not support the video tag.
      </video>
    );
  }

  return (
    <div>

    </div>
  )
}
