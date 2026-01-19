/* stb_image - v2.28 - public domain image loader - http://nothings.org/stb
   Minimal implementation for basic image loading support
*/

#ifndef STBI_INCLUDE_STB_IMAGE_H
#define STBI_INCLUDE_STB_IMAGE_H

#ifndef STBIDEF
#define STBIDEF extern
#endif

#ifdef __cplusplus
extern "C" {
#endif

STBIDEF unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
STBIDEF void stbi_image_free(void *retval_from_stbi_load);
STBIDEF const char *stbi_failure_reason(void);

#ifdef __cplusplus
}
#endif

#endif // STBI_INCLUDE_STB_IMAGE_H

#ifdef STB_IMAGE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const char *stbi__g_failure_reason;

STBIDEF const char *stbi_failure_reason(void) {
    return stbi__g_failure_reason;
}

static void *stbi__malloc(size_t size) {
    return malloc(size);
}

STBIDEF void stbi_image_free(void *retval_from_stbi_load) {
    free(retval_from_stbi_load);
}

// Simple BMP/TGA/PNG loading - for production use, download full stb_image.h from:
// https://github.com/nothings/stb/blob/master/stb_image.h

// Minimal JPEG decoder using system libraries on macOS
#ifdef __APPLE__
#include <ImageIO/ImageIO.h>
#include <CoreFoundation/CoreFoundation.h>

STBIDEF unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels) {
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8*)filename, strlen(filename), false);
    if (!url) {
        stbi__g_failure_reason = "Failed to create URL";
        return NULL;
    }

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
    CFRelease(url);
    if (!source) {
        stbi__g_failure_reason = "Failed to create image source";
        return NULL;
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CFRelease(source);
    if (!image) {
        stbi__g_failure_reason = "Failed to create image";
        return NULL;
    }

    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);

    *x = (int)width;
    *y = (int)height;
    *channels_in_file = 4; // We always decode to RGBA

    int out_channels = desired_channels == 0 ? 4 : desired_channels;

    unsigned char *data = (unsigned char*)stbi__malloc(width * height * out_channels);
    if (!data) {
        CGImageRelease(image);
        stbi__g_failure_reason = "Out of memory";
        return NULL;
    }

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast;

    if (out_channels == 3) {
        bitmapInfo = kCGImageAlphaNoneSkipLast;
    } else if (out_channels == 1) {
        CGColorSpaceRelease(colorSpace);
        colorSpace = CGColorSpaceCreateDeviceGray();
        bitmapInfo = kCGImageAlphaNone;
    }

    CGContextRef context = CGBitmapContextCreate(data, width, height, 8,
        width * out_channels, colorSpace, bitmapInfo);
    CGColorSpaceRelease(colorSpace);

    if (!context) {
        free(data);
        CGImageRelease(image);
        stbi__g_failure_reason = "Failed to create bitmap context";
        return NULL;
    }

    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CGImageRelease(image);

    // Unpremultiply alpha if needed
    if (out_channels == 4) {
        for (size_t i = 0; i < width * height; i++) {
            unsigned char *pixel = data + i * 4;
            unsigned char a = pixel[3];
            if (a > 0 && a < 255) {
                pixel[0] = (unsigned char)((pixel[0] * 255) / a);
                pixel[1] = (unsigned char)((pixel[1] * 255) / a);
                pixel[2] = (unsigned char)((pixel[2] * 255) / a);
            }
        }
    }

    return data;
}

#else
// Fallback for other platforms - returns NULL
STBIDEF unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels) {
    (void)filename; (void)x; (void)y; (void)channels_in_file; (void)desired_channels;
    stbi__g_failure_reason = "stb_image: Please download full implementation from https://github.com/nothings/stb";
    return NULL;
}
#endif

#endif // STB_IMAGE_IMPLEMENTATION
