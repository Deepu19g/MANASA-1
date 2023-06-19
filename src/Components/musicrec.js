import { useEffect, useState } from "react";
import "./musicrec.css";
import Navbar from "./Navbar";
import Footer from "./Footer";

export default function MusicRecs() {
  // const [emotion, setEmotion] = useState();
  const [songs, setSongs] = useState([]);

  const firstcall = async () => {
    var dataR = await (await fetch("http://localhost:5000/members")).text();
    //setEmotion(dataR)
    console.log(dataR);
    fetch(`http://localhost:5000/music?emotion=${dataR}`)
      .then((response) => response.json())
      .then((data) => {
        setSongs(data.songs);
        console.log(data.songs); // Prints the array of movies to the console
      })
      .catch((error) => console.error(error));
  };
  useEffect(() => {
    /* fetch("/members").then(
      data => {
        console.log(data)
        fetch(`http://localhost:5000/movies?emotion=${data}`)
    
    .then(data => {
      setMovies(data.movies)
      console.log(data); // Prints the array of movies to the console
    })
    .catch(error => console.error(error));
      }
    )*/
    firstcall();
  }, []);
  return (
    <>
    <div className="music-page">
      <Navbar />
      <div >
        <h1>Music Recommendations for your mood</h1>
        <ul>
          {songs && songs.map((song) => {
            var src_string = `https://open.spotify.com/embed/track/${song['spotify_id']}?utm_source=generator`
            console.log(src_string);
            return (
              <iframe title="title" src={src_string} width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
            );
          })}
        </ul>
      </div>
      <Footer />
    </div>
    </>
  );
}
