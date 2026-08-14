"use client";

import { useEffect } from "react";

export default function ClearAuthAndRedirect() {
  useEffect(() => {
    // Clear cookies and local storage
    document.cookie = "budai_token=; path=/; max-age=0";
    localStorage.removeItem("budai_token");
    localStorage.removeItem("budai_user_name");
    
    // Force a full reload to /login so the middleware sees the cleared cookie
    window.location.href = "/login";
  }, []);

  return null;
}
