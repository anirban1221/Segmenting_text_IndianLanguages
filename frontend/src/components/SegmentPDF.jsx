import React, { useState } from 'react';
import axios from 'axios';

const SegmentPDF = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [segments, setSegments] = useState([]);
  const [error, setError] = useState(null); 
  const [loading, setLoading] = useState(false);

  const containerStyle = "min-h-screen flex flex-col items-center justify-center bg-gray-100 dark:bg-gray-900 p-6 text-gray-800 dark:text-white mt-6";
  const columnContainerStyle = "w-full max-w-5xl grid grid-cols-1 md:grid-cols-2 gap-6";
  const fileUploadContainerStyle = "flex flex-col";
  const fileUploadLabelStyle = "flex flex-col items-center justify-center w-full h-40 px-4 transition bg-white dark:bg-gray-800 border-2 border-dashed rounded-md cursor-pointer dark:border-gray-600 hover:border-blue-400 hover:bg-gray-50 dark:hover:border-blue-300 dark:hover:bg-gray-700";
  const fileNameStyle = "mt-2 text-sm text-green-600 dark:text-green-400";
  const buttonStyle = "mt-4 px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700";
  const errorMessageStyle = "mt-2 text-sm text-red-600 dark:text-red-400";
  const outputContainerStyle = "w-full h-64 p-4 border rounded overflow-y-auto bg-white dark:bg-gray-800 dark:border-gray-600 text-gray-800 dark:text-white";
  const segmentListStyle = "list-disc pl-5";
  const outputTextStyle = "italic text-gray-500";

  const handleFileChange = (e) => {
    setPdfFile(e.target.files[0]);
    setError(null); 
  };

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!pdfFile) return alert("Please select a PDF file!");

    const formData = new FormData();
    formData.append('file', pdfFile);

    setLoading(true); 
    setError(null); 

    try {
        const response = await axios.post('http://localhost:8000/segment-pdf', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        if (response.data && response.data.segments) {
            setSegments(response.data.segments);
        } else {
            setError("Unexpected response format.");
        }
    } catch (err) {
        console.error(err); 
        setError("Something went wrong. Please try again later.");
    } finally {
        setLoading(false); 
    }
};

  return (
    <div className={containerStyle}>
      <div className={columnContainerStyle}>

        {/* Left Column: File Upload */}
        <div className={fileUploadContainerStyle}>
          <h2 className="text-xl font-bold mb-2">Upload a PDF</h2>

          <label className={fileUploadLabelStyle}>
            <span className="text-sm text-gray-600 dark:text-gray-300">Click to select a PDF</span>
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {pdfFile && (
            <p className={fileNameStyle}>
              Selected: {pdfFile.name}
            </p>
          )}

          <button
            onClick={handleUpload}
            className={buttonStyle}
            disabled={loading} // Disable button during upload
          >
            {loading ? 'Uploading...' : 'Upload & Segment'}
          </button>

          {/* Error message */}
          {error && (
            <p className={errorMessageStyle}>
              {error}
            </p>
          )}
        </div>

        {/* Right Column: Output */}
        <div>
          <h2 className="text-xl font-bold mb-2">Segmented Output</h2>
          <div className={outputContainerStyle}>
            {segments.length > 0 ? (
              <ul className={segmentListStyle}>
                {segments.map((segment, index) => (
                  <li key={index} className="mb-2">{segment}</li>
                ))}
              </ul>
            ) : (
              <p className={outputTextStyle}>Output will appear here.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SegmentPDF;
