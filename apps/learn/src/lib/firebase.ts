import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyBT4YE-hAThkisCtf4jSerQiJFT9cS71ok",
  authDomain: "unbox-ai-project.firebaseapp.com",
  projectId: "unbox-ai-project",
  storageBucket: "unbox-ai-project.firebasestorage.app",
  messagingSenderId: "264603738016",
  appId: "1:264603738016:web:a4bc3cdaffe90e2b498ec5",
  measurementId: "G-RPSEQT3EK6",
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db = getFirestore(app);
export const googleProvider = new GoogleAuthProvider();
