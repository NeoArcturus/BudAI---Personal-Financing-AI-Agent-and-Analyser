import Image from "next/image";

export default function GlobalBackground() {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden select-none bg-black">
      <Image
        src="/background.jpg"
        alt=""
        fill
        priority
        quality={90}
        className="object-cover object-center"
      />
      {/* Ambient overlay ensuring crisp contrast and institutional readability */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/20 to-black/60 pointer-events-none" />
    </div>
  );
}
