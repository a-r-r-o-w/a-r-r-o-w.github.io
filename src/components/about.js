export default function About({description, additional_info}) {
  return (
    <div>
      <span className="heading">about</span>
      
      <p className="mt-2 text-xl">
        {description}
      </p>
      <p className="pt-8 text-sm">
        {additional_info}
      </p>
    </div>
  )
}
