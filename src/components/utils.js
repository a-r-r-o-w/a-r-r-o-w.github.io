const missingGradientImage = (alt = "missing-image", index = -1) => {
  return (
    <img
      // alt={alt}
      className="missing-image-gradient cards__img"
      style={{"--gradient-angle": index % 2 === 0 ? '45deg' : '135deg'}}
    />
  )
};

export default missingGradientImage;
