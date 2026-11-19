import path from "path";

const projectNodeModules = path.resolve("./node_modules");
if (!process.env.NODE_PATH) {
  process.env.NODE_PATH = projectNodeModules;
} else if (!process.env.NODE_PATH.includes(projectNodeModules)) {
  process.env.NODE_PATH = `${projectNodeModules}:${process.env.NODE_PATH}`;
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(process.env.VERCEL
    ? {}
    : {
        turbopack: {
          root: path.resolve("."),
        },
      }),
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8080",
        pathname: "/api/media/image/**",
      },
      {
        protocol: "https",
        hostname: "**",
      },
      {
        protocol: "http",
        hostname: "**",
      },
    ],
  },
};

export default nextConfig;
