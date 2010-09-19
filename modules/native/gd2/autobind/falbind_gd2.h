// Special settings and changes to GD2 header files
// to work with falbind

// Synthetic GD macros as functions.
#include "gd.h"

#undef gdImageAlpha
#undef gdImageBlue
#undef gdImageGreen
#undef gdImageRed
#undef gdImageSX
#undef gdImageSY
#undef gdImageTrueColor

int gdImageAlpha(gdImage* im, int color);
int gdImageBlue(gdImage* im, int color);
int gdImageGreen(gdImage* im, int color);
int gdImageRed(gdImage* im, int color);

int gdImageSX(gdImage* im);
int gdImageSY(gdImage* im);
int gdImageTrueColor(gdImage* im);

gdFont* gdFontGetTiny();
gdFont* gdFontGetSmall();
gdFont* gdFontGetMediumBold();
gdFont* gdFontGetLarge();
gdFont* gdFontGetGiant();
