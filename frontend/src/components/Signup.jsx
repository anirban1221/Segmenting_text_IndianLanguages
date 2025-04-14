import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth } from '../firebase';
import {
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
} from 'firebase/auth';
import { FcGoogle } from 'react-icons/fc';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const container = "min-h-screen flex items-center justify-center bg-gradient-to-r from-blue-100 via-white to-pink-100 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 transition-all duration-300 px-4";
const card = "bg-white dark:bg-gray-800 rounded-3xl shadow-2xl p-10 w-full max-w-md";
const heading = "text-3xl font-bold text-center mb-6 text-blue-700 dark:text-white";
const inputField = "w-full p-3 rounded-xl border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-blue-400 outline-none transition-all duration-200";
const submitBtn = "w-full bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 transition-all font-semibold";
const googleBtn = "flex items-center justify-center gap-2 w-full p-3 border rounded-xl dark:bg-gray-700 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-600 transition-all";
const orText = "block mb-2 text-gray-500 dark:text-gray-400 text-center";

const Signup = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSignup = async (e) => {
    e.preventDefault();
    try {
      await createUserWithEmailAndPassword(auth, email, password);
      toast.success("Account created successfully!");
      navigate('/');
    } catch (err) {
      if (err.message.includes("email-already-in-use")) {
        toast.error("This email is already in use.");
      } else if (err.message.includes("weak-password")) {
        toast.error("Password should be at least 6 characters.");
      } else {
        toast.error("Signup failed. Please try again.");
      }
    }
  };

  const handleGoogleSignup = async () => {
    const provider = new GoogleAuthProvider();
    try {
      await signInWithPopup(auth, provider);
      toast.success("Signed up with Google!");
      navigate('/');
    } catch (err) {
      toast.error("Google signup failed. Please try again.");
    }
  };

  return (
    <div className={container}>
      <div className={card}>
        <h2 className={heading}>Create an Account ✨</h2>
        <form onSubmit={handleSignup} className="space-y-4">
          <input
            type="email"
            placeholder="Enter your email"
            className={inputField}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Create a password"
            className={inputField}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="submit" className={submitBtn}>Sign Up</button>
        </form>

        <div className="mt-6">
          <span className={orText}>Or sign up with</span>
          <button onClick={handleGoogleSignup} className={googleBtn}>
            <FcGoogle className="text-2xl" />
            <span className="text-gray-700 dark:text-white font-medium">Continue with Google</span>
          </button>
        </div>
      </div>

      <ToastContainer position="top-right" autoClose={3000} hideProgressBar />
    </div>
  );
};

export default Signup;
