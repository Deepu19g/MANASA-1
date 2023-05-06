import { Container, Row, Col } from "react-bootstrap";
import { Routes, Route } from "react-router-dom";
import "./App.css"; 
import Home from "./Components/Home";
import Login from "./Components/Login";
import Signup from "./Components/Signup";
import ProtectedRoute from "./Components/ProtectedRoute";
import { UserAuthContextProvider } from "./context/UserAuthContext";
import Footer from "./Components/Footer";
import MovieRecc from "./Components/movierecc";
import MusicRecs from "./Components/musicrec";
import Mindful from "./Components/Mindful";

function App() {
 
  return (
    <div className="App">

          <UserAuthContextProvider>
            <Routes>
              <Route
                path="/Home"
                element={
                  <ProtectedRoute>
                    <Home />
                  </ProtectedRoute>
                }
              />
              <Route path="/" element={<Login />} />
              <Route path="/Signup" element={<Signup />} />
              <Route path="/MovieRecc" element={<MovieRecc />} />
              <Route path="/musicrec" element={<MusicRecs />} />
              <Route path="/Mindful" element={<Mindful />} />
              
              
            </Routes>
          </UserAuthContextProvider>
     </div>
      
  
  );
}

export default App;
