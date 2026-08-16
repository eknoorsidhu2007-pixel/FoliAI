import ImageUpload from "./components/ImageUpload";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center bg-zinc-50 px-6 py-16 font-sans">
      <main className="flex w-full max-w-2xl flex-col gap-8">
        <header className="text-center sm:text-left">
          <p className="text-sm font-medium tracking-wide text-emerald-700">
            FoliAI
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-900">
            Plant disease classifier
          </h1>
          <p className="mt-2 text-base text-zinc-600">
            Upload a leaf photo to get a predicted class and confidence score.
          </p>
        </header>

        <ImageUpload />
      </main>
    </div>
  );
}
