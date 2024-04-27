"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Image from "next/image";
import missingGradientImage from "./utils";

export default function ProjectCard({url, name, description, languages, image, video, index}) {
  let [canPlayVideo, setCanPlayVideo] = useState(false);
  const videoRef = useRef(null);

  console.log("langs", languages)
  
  useEffect(() => {
    setCanPlayVideo(document.createElement("video").canPlayType("video/mp4") !== "");
  }, []);
  
  let displayContent = missingGradientImage(name, index);

  if (image !== null)
    displayContent = <Image
      className="cards__img"
      src={image}
      alt={name}
      width={300}
      height={300}
    />;
  else if (video !== null) {
    if (video.endsWith("gif"))
      displayContent = <Image
        className="cards__img"
        src={video}
        alt={name}
        width={300}
        height={300}
      />;
    else if (canPlayVideo)
      displayContent = (
        <video
          className="cards__vid"
          ref={videoRef}
          loop muted playsInline
          alt={name}
        >
          {
            video.endsWith("mp4") ?
              <>
                <source src={video} type="video/mp4" />
              </> :
              <>
                <source src={video + ".webm"} type="video/webm" />
                <source src={video + ".mp4"} type="video/mp4" />
              </>
          }

          Your browser does not support the video tag.
        </video>
      );
  }

  return (
    <div
      className="cards__project"
      onMouseEnter={() => {
        if (videoRef.current !== null)
          videoRef.current.play();
      }}
      onMouseLeave={() => {
        if (videoRef.current != null) {
          videoRef.current.pause();
          videoRef.current.currentTime = 0;
        }
      }}
    >
      <a href={url}>
        {/* {displayContent} */}
        <Suspense fallback={missingGradientImage(index)}>
          {displayContent}
        </Suspense>
        <div className="cards__text">
          <div className="font-bold text-xl;">{name}</div>
          <div className="text-base" id="description">{description}</div>
          <div className="cards__languages">
            {
              languages.map((language, index) => {
                return (
                  <span className="cards__language_container" key={index}>{language}</span>
                );
              })
            }
          </div>
        </div>
      </a>
    </div>
  )
}
