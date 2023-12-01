from anime2sd import download_images


if __name__ == "__main__":
    output_dir = "data/intermediate/16bit/booru/raw"
    tags = ["16bit_sensation"]
    limit_per_character = 6
    max_image_size = 640
    character_mapping = {
        "akisato_konoha": "Konoha",
        "riko_(machikado_mazoku)": "Riko",
        "riko_(made_in_abyss)": "",
    }
    save_aux = ["tags", "characters"]

    # Call the function
    download_images(
        output_dir=output_dir,
        tags=tags,
        limit_per_character=limit_per_character,
        max_image_size=max_image_size,
        character_mapping=character_mapping,
        save_aux=save_aux,
    )
