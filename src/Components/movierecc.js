import { useState, useEffect } from "react";
import "./movierecc.css";
import Navbar from "./Navbar.js";
import Footer from "./Footer.js";

export default function MovieRecc() {
  // const [emotion, setEmotion] = useState();
  const [movies, setMovies] = useState([]);

  const firstcall = async () => {
    var dataR = await (await fetch("http://localhost:5000/members")).text();
    //setEmotion(dataR)
    console.log(dataR);
    fetch(`http://localhost:5000/movies?emotion=${dataR}`)
      .then((response) => response.json())
      .then((data) => {
        setMovies(data.movies);
        console.log(data.movies); // Prints the array of movies to the console
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
    
    <div
      style={{
        // backgroundImage: `url(${backgroundImg})`,
        backgroundSize: "111%",
        minHeight: "100vh",
        minWidth: "100vw",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        /*marginLeft:0*/
      }}
    >
    <Navbar/>

      <h1
        className="text-3xl "
        style={{
          textAlign: "center",
          fontSize: "3rem",
          color: "black",
        }}
      >
        Movie Recommendation System
      </h1>

      <div
        style={{ textAlign: "center", marginTop: "2rem", fontSize: "1.2rem" }}
      >
        <button
          onClick={firstcall}
          style={{
            background: "#6385dc",
            borderRadius: "0.5rem",
            padding: "0.5rem",
            color: "white",
            cursor: "pointer",
          }}
        >
          Find Movies
        </button>
      </div>
      <br></br>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
          marginTop: "2rem",
          margin: "auto",
          fontSize: "1rem",
          fontFamily: "cursive",
        }}
      >
        {movies.map((movie, id) => {
          return (
            <div
              key={id}
              className="animate__animated animate__fadeInUp"
              style={{
                animationDelay: `${id * 0.1}s`,
                backgroundColor: "#f2f2f2",
                padding: "1rem",
                transition: "background-color 0.3s ease",
                border: "1px solid #ccc",
                borderRadius: "0.5rem",
                boxShadow: "2px 2px 5px rgba(0, 0, 0, 0.3)",
                marginBottom: "1rem",
                width: "700px",
                height: "175px",
              }}
            >
              <a
                href={movie.imdb_link}
                target="_blank"
                rel="noreferrer"
                style={{ color: "blue" }}
              >
                <h3 style={{ margin: "0" }}>
                  {movie.title}
                  {movie.year}
                </h3>
              </a>
              <p style={{ margin: "0.5rem 0" }}>Rating: {movie.rating}</p>
              <p style={{ margin: "0.5rem 0" }}>Genre: {movie.genre}</p>
              <p style={{ margin: "0.5rem 0" }}>Cast: {movie.cast}</p>
              <p style={{ margin: "0.5rem 0" }}>Director: {movie.director}</p>
            </div>
          );
        })}
        <Footer/>
      </div>
    </div>
  );
}
