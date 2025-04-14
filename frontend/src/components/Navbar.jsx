import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FiSun, FiMoon } from 'react-icons/fi';
import { FaUserPlus, FaSignInAlt } from 'react-icons/fa';
import logo from '../assets/logo.png';
const navBase = "w-full fixed top-0 left-0 z-50 transition-all duration-300 bg-gradient-to-r from-pink-500 via-orange-500 to-yellow-500 text-white dark:bg-gradient-to-r dark:from-gray-900 dark:via-gray-800 dark:to-gray-900";
const navContainer = "max-w-7xl mx-auto px-6 py-4 flex justify-between items-center";

const logoWrapper = "flex items-center gap-4 cursor-pointer hover:scale-105 transition-transform";
const logoContainer = "h-14 w-14 rounded-full overflow-hidden flex items-center justify-center shadow-lg ring-2 ring-white";
const logoText = "text-3xl font-extrabold tracking-wide font-sans text-white dark:text-white";

const actionWrapper = "flex items-center gap-6";
const toggleBtn = "text-2xl p-2 rounded-full bg-white/20 hover:bg-white/30 dark:hover:bg-white/10 transition backdrop-blur-sm";
const loginBtn = "flex items-center gap-3 px-6 py-3 border-2 border-white text-white rounded-full hover:bg-white/20 dark:hover:bg-white/10 transition font-semibold shadow-lg backdrop-blur-sm";
const signupBtn = "flex items-center gap-3 px-6 py-3 bg-white text-blue-600 rounded-full hover:bg-blue-100 transition font-semibold shadow-md";

const Navbar = () => {
  const [darkMode, setDarkMode] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setDarkMode(prefersDark);
    if (prefersDark) {
      document.documentElement.classList.add('dark');
    }
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
    document.documentElement.classList.toggle('dark');
  };

  const handleLogoClick = () => navigate('/');

  return (
    <nav className={navBase}>
      <div className={navContainer}>
        {/* Logo */}
        <div className={logoWrapper} onClick={handleLogoClick}>
          <div className={logoContainer}>
            <img src={logo} alt="Vakya" className="h-full w-full object-cover" />
          </div>
          <h1 className={logoText}>Vakya</h1>
        </div>

        {/* Right section */}
        <div className={actionWrapper}>
          <button
            onClick={toggleDarkMode}
            className={toggleBtn}
            aria-label="Toggle Dark Mode"
          >
            {darkMode ? <FiSun className="text-yellow-400" /> : <FiMoon className="text-white" />}
          </button>

          <Link to="/login">
            <button className={loginBtn}>
              <FaSignInAlt />
              <span className="hidden sm:inline">Login</span>
            </button>
          </Link>

          <Link to="/signup">
            <button className={signupBtn}>
              <FaUserPlus />
              <span className="hidden sm:inline">Signup</span>
            </button>
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
