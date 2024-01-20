import { useEffect, useState } from "react";
import Link from "next/link";

export default function Header({name, setMode}) {
  return (
    <header>
      <nav>
        <a href="/" className="name">{name}</a>

        <div className="flex justify-between items-center space-x-4 mx-2">
          <Link className="navlink" href="/">about</Link>
          <Link className="navlink" href="/projects/">projects</Link>
          <Link className="navlink" href="/">blog</Link>
          <Link className="navlink" href="https://drive.google.com/file/d/1A53ZjSHKMQQxEl6R5EmiEUSYJK4_dA0x/view" target="_blank">cv</Link>
          
          <div className="toggle-button" onClick={() => {
            setMode((prevMode) => {
              const newMode = prevMode === "dark-mode" ? "light-mode" : "dark-mode";
              return newMode;
            });
          }}>
            <div className="toggle-icon">
              <div className="moon-icon">🌙</div>
              <div className="sun-icon">☀️</div>
            </div>
            <div className="toggle-switch"></div>
          </div>
        </div>
      </nav>
    </header>
  );
}