"use client";

import { useState, useEffect } from "react";
import { Inter } from "next/font/google";

import "../styles/globals.css";
import Header from "@/components/header";
import Footer from "@/components/footer";


const inter = Inter({ subsets: ["latin"] });

// export const metadata = {
//   title: "Aryan V S",
//   description: "Aryan"s Portfolio Site",
// };

export default function RootLayout({ children }) {
  const [mode, setMode] = useState("dark-mode");

  useEffect(() => {
    const theme = localStorage.getItem("theme");
    if (theme === "light-mode") {
      setMode("light-mode");
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("theme", mode);
    document.body.classList.toggle("dark-mode", mode === "dark-mode");
    document.body.classList.toggle("light-mode", mode === "light-mode");
  }, [mode]);
  
  return (
    <html lang="en">
      <body className={`${inter.className} ${mode}`}>
        {/* <!-- Header --> */}
        <Header
          name="Aryan V S"
          setMode={setMode}
        />
        
        {children}

        <Footer />
      </body>
    </html>
  )
}
