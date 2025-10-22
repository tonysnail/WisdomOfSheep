export default function JsonViewer({ value }: { value: any }) {
  return (
    <pre className="json">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}