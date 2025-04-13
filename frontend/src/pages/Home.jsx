import React from 'react';
import { Link } from 'react-router-dom';
import { getAuth } from 'firebase/auth';

const wrapper = "min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-white flex flex-col justify-center items-center px-4 py-10";
const container = "max-w-2xl text-center space-y-6";
const heading = "text-4xl font-bold";
const subText = "text-lg text-gray-600 dark:text-gray-300";
const buttonGroup = "flex flex-col sm:flex-row gap-4 justify-center mt-8";
const btnPrimary = "px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 transition focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50";
const btnPdf = "px-6 py-3 bg-green-600 text-white rounded hover:bg-green-700 transition focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-opacity-50";
const btnLogin = "px-6 py-3 border border-gray-400 text-gray-700 dark:text-white rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50";

const Home = () => {
  const auth = getAuth();
  const user = auth.currentUser;

  return (
    <div className={wrapper}>
      <div className={container}>
        <h1 className={heading}>
          Welcome{user ? `, ${user.displayName || 'Valued User'}` : ''} 👋
        </h1>

        <p className={subText}>
          Segment Indian language text with gender-aware, context-rich logic. Start with free text or upload a PDF (if logged in).
        </p>

        <div className={buttonGroup}>
          <Link to="/segmenter">
            <button 
              className={btnPrimary} 
              title="Click to segment text" 
              aria-label="Segment Text"
            >
              Segment Text
            </button>
          </Link>

          {user ? (
            <Link to="/segmenter/pdf">
              <button 
                className={btnPdf} 
                title="Click to upload a PDF for segmentation" 
                aria-label="Upload PDF"
              >
                Upload PDF
              </button>
            </Link>
          ) : (
            <Link to="/login">
              <button 
                className={btnLogin} 
                title="Login to upload PDF" 
                aria-label="Login to Upload PDF"
              >
                Login to Upload PDF
              </button>
            </Link>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;
