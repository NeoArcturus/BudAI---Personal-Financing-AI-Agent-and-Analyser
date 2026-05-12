"use client";
import { redirect } from "next/navigation";

const Page = () => {
  if (!localStorage.getItem("budai_token")) {
    redirect("/login");
  }
  redirect("/home");
};

export default Page;
