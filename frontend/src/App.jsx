import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import SegmentText from './components/SegmentText';
import SegmentPDF from './components/SegmentPDF';
import Login from './components/Login';
import Signup from './components/Signup';

function App() {
  return (
    <Router>
      <div className="bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-white min-h-screen">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/segmenter" element={<SegmentText />} />
          <Route path="/segmenter/pdf" element={<SegmentPDF />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
