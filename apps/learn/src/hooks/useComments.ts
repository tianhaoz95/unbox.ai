import { useEffect, useState } from "react";
import {
  collection,
  query,
  where,
  orderBy,
  onSnapshot,
  addDoc,
  serverTimestamp,
} from "firebase/firestore";
import { db } from "../lib/firebase";
import { useAuth } from "../contexts/AuthContext";

export interface Comment {
  id: string;
  chapterId: string;
  selectedText: string;
  anchorText: string; // ~60 chars of surrounding context for disambiguation
  comment: string;
  userId: string;
  userDisplayName: string;
  userPhotoURL: string;
  createdAt: Date;
  resolved: boolean;
}

export function useComments(chapterId: string) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();

  useEffect(() => {
    const q = query(
      collection(db, "comments"),
      where("chapterId", "==", chapterId),
      where("resolved", "==", false),
      orderBy("createdAt", "desc")
    );

    const unsub = onSnapshot(q, (snap) => {
      setComments(
        snap.docs.map((doc) => ({
          id: doc.id,
          ...(doc.data() as Omit<Comment, "id" | "createdAt">),
          createdAt: doc.data().createdAt?.toDate() ?? new Date(),
        }))
      );
      setLoading(false);
    });

    return unsub;
  }, [chapterId]);

  const addComment = async (
    selectedText: string,
    anchorText: string,
    comment: string
  ) => {
    if (!user) throw new Error("Must be signed in to comment");
    await addDoc(collection(db, "comments"), {
      chapterId,
      selectedText,
      anchorText,
      comment,
      userId: user.uid,
      userDisplayName: user.displayName ?? "Anonymous",
      userPhotoURL: user.photoURL ?? "",
      createdAt: serverTimestamp(),
      resolved: false,
    });
  };

  return { comments, loading, addComment };
}
