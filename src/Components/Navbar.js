/* eslint-disable jsx-a11y/anchor-is-valid */
import React, { useState } from "react";
import Logo from "../Assets/Logo.png";
import { HiOutlineBars3 } from "react-icons/hi2";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import HomeIcon from "@mui/icons-material/Home";
import LiveTvIcon from "@mui/icons-material/LiveTv";
import PsychologyIcon from "@mui/icons-material/Psychology";
import LibraryBooksIcon from "@mui/icons-material/LibraryBooks";
import HelpIcon from "@mui/icons-material/Help";

const Navbar = () => {
  const [openMenu, setOpenMenu] = useState(false);
  const menuOptions = [
    {
      text: "Home",
      icon: <HomeIcon />,
    },
    {
      text: "Recommendation",
      icon: <LiveTvIcon />,
    },
    {
      text: "MindfulExercises",
      icon: <PsychologyIcon />,
    },
    {
      text: "Articles",
      icon: <LibraryBooksIcon />,
    },
    {
      text: "Help",
      icon: <HelpIcon />,
    },
  ];
  return (
    <nav>
      <div className="nav-logo-container">
        <a href="/home">
          <img className="logo" src={Logo} alt="" />
        </a>
      </div>
      <div className="navbar-links-container">
        <a href="/MovieRecc">Movie Recommendations </a>

        <a href="/musicrec">Music Recommendations</a>

        
        <a href="/Mindful">MindfulExercises</a>
        <a href="/"><button className="primary-button">Log Out</button></a>
      </div>
      <div className="navbar-menu-container">
        <HiOutlineBars3 onClick={() => setOpenMenu(true)} />
      </div>
      <Drawer open={openMenu} onClose={() => setOpenMenu(false)} anchor="right">
        <Box
          sx={{ width: 250 }}
          role="presentation"
          onClick={() => setOpenMenu(false)}
          onKeyDown={() => setOpenMenu(false)}
        >
          <List>
            {menuOptions.map((item) => (
              <ListItem key={item.text} disablePadding>
                <ListItemButton>
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
          <Divider />
        </Box>
      </Drawer>
    </nav>
  );
};

export default Navbar;
