import React, { useState } from 'react';
import axios from 'axios';

const wrapper = "min-h-screen pt-32 flex flex-col items-center justify-start bg-gray-100 dark:bg-gray-900 p-6 text-gray-800 dark:text-white";
const container = "w-full max-w-5xl flex flex-col gap-6";

const languageContainer = "flex flex-col md:flex-row justify-start items-start md:items-center gap-4";
const languageLabel = "text-sm font-medium";
const languageSelect = "p-2 w-36 border border-gray-300 rounded bg-white shadow-sm dark:bg-gray-800 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none";

const contentGrid = "grid grid-cols-1 md:grid-cols-2 gap-6";

const inputBoxWrapper = "flex flex-col h-[28rem] justify-between";
const heading = "text-xl font-bold mb-2";
const textArea = "w-full h-80 p-4 border border-gray-300 rounded bg-white shadow-md dark:bg-gray-800 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:outline-none";
const segmentBtn = "mt-2 px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400";


const outputBoxWrapper = "flex flex-col h-[28rem]";
const outputBox = "w-full h-80 p-4 border border-gray-300 rounded bg-white shadow-md overflow-y-auto dark:bg-gray-800 dark:border-gray-600 max-h-80";
const segmentList = "list-disc pl-5";
const placeholderText = "italic text-gray-500";
const loadingText = "text-yellow-500 italic";

const SegmentText = () => {
  const [inputText, setInputText] = useState('');
  const [segments, setSegments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [language, setLanguage] = useState('Hindi');

  const handleSegment = async () => {
    if (inputText.trim() === '') return;
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/segment-text', {
        text: inputText,
        language: language
      });
      setSegments(response.data.segments || []);
    } catch (error) {
      console.error('Error segmenting text:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={wrapper}>
      <div className={container}>

        {/* Language Select Container */}
        <div className={languageContainer}>
          <label htmlFor="language" className={languageLabel}>Select Language</label>
          <select
            id="language"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className={languageSelect}
            aria-label="Select language for segmentation"
          >
            <option value="Hindi">Hindi</option>
          </select>
        </div>

        {/* Grid for Input and Output */}
        <div className={contentGrid}>

          {/* Left Column: Input */}
          <div className={inputBoxWrapper}>
            <div>
              <h2 className={heading}>Enter Text</h2>
              <textarea
                className={textArea}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={`Type or paste your ${language} text here...`}
                aria-label={`Input ${language} text to segment`}
              />
            </div>
            <button onClick={handleSegment} className={segmentBtn} aria-label="Segment the entered text">
              {isLoading ? 'Segmenting...' : 'Segment Text'}
            </button>
          </div>

          {/* Right Column: Output */}
          <div className={outputBoxWrapper}>
            <h2 className={heading}>Segmented Output</h2>
            <div className={outputBox}>
              {isLoading ? (
                <p className={loadingText}>Segmenting text, please wait...</p>
              ) : segments.length > 0 ? (
                <ul className={segmentList}>
                  {segments.map((segment, index) => (
                    <li key={index} className="mb-2">{segment}</li>
                  ))}
                </ul>
              ) : (
                <p className={placeholderText}>Output will appear here.</p>
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default SegmentText;
