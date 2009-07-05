extern "C" {
   #include <gd.h>
}

#include <falcon/engine.h>
#include <falcon/autocstring.h>
#include <falcon/autowstring.h>

using namespace Falcon;


class _falbind_GdFont: public CoreObject
{
public:
   _falbind_GdFont(const CoreClass* cgen, void* ud, bool):
      CoreObject( cgen )
   {
      setUserData( ud );
   }
   
   ~_falbind_GdFont() {
      
   }

   virtual bool setProperty( const String &prop, const Item &value )
   {
      return false;
   }


   virtual bool getProperty( const String &key, Item &ret ) const
   {
      return false;
   }
   
   // use our cloner
   virtual CoreObject* clone() const {
      return 0;
   }
};

static CoreObject* _falbind_GdFont_factory( const CoreClass* cgen, void* ud, bool bDeser )
{
   if ( ud == 0 )
      throw new CodeError( ErrorParam( e_non_callable, __LINE__ )
         .extra( "Opaque class instantiated" ) );
         
   return new _falbind_GdFont(cgen, ud, bDeser);
}


class _falbind_GdFont: public CoreObject
{
public:
   _falbind_GdFont(const CoreClass* cgen, void* ud, bool):
      CoreObject( cgen )
   {
      setUserData( ud );
   }
   
   ~_falbind_GdFont() {
      
   }

   virtual bool setProperty( const String &prop, const Item &value )
   {
      return false;
   }


   virtual bool getProperty( const String &key, Item &ret ) const
   {
      return false;
   }
   
   // use our cloner
   virtual CoreObject* clone() const {
      return 0;
   }
};

static CoreObject* _falbind_GdFont_factory( const CoreClass* cgen, void* ud, bool bDeser )
{
   if ( ud == 0 )
      throw new CodeError( ErrorParam( e_non_callable, __LINE__ )
         .extra( "Opaque class instantiated" ) );
         
   return new _falbind_GdFont(cgen, ud, bDeser);
}


class _falbind_gdFTStringExtra$: public CoreObject
{
public:
   _falbind_gdFTStringExtra$(const CoreClass* cgen, void* ud, bool):
      CoreObject( cgen )
   {
      setUserData( ud );
   }
   
   ~_falbind_gdFTStringExtra$() {
      
   }

   virtual bool setProperty( const String &prop, const Item &value )
   {
      return false;
   }


   virtual bool getProperty( const String &key, Item &ret ) const
   {
      return false;
   }
   
   // use our cloner
   virtual CoreObject* clone() const {
      return 0;
   }
};

static CoreObject* _falbind_gdFTStringExtra$_factory( const CoreClass* cgen, void* ud, bool bDeser )
{
   if ( ud == 0 )
      throw new CodeError( ErrorParam( e_non_callable, __LINE__ )
         .extra( "Opaque class instantiated" ) );
         
   return new _falbind_gdFTStringExtra$(cgen, ud, bDeser);
}


class _falbind_GdImage: public FalconObject
{
public:
   _falbind_GdImage(const CoreClass* cgen, void* ud, bool bd ):
      FalconObject( cgen, bd )
   {
      setUserData( ud );
   }

   ~_falbind_GdImage() {
      gdImageDestroy( this->get() );
   }

   // use our cloner
   virtual CoreObject* clone() const {
      return 0;
   }

   gdImage* get() const { return static_cast<gdImage*>( this->getUserData() ); }
};

static CoreObject* _falbind_GdImage_factory( const CoreClass* cgen, void* ud, bool bDeser )
{
   return new _falbind_GdImage(cgen, ud, bDeser);
}


static void _falbind_gdImageCreateFromJpegPtr( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isMemBuf()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,M" ) );
   }

int size = (int) vm->param(0)->forceInteger();
void* data = (void*) vm->param(1)->asMemBuf()->data();

   gdImageStruct* __funcreturn__ = gdImageCreateFromJpegPtr( size, data );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_CreateTrueColor( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

int sx = (int) vm->param(0)->forceInteger();
int sy = (int) vm->param(1)->forceInteger();

   gdImageStruct* __funcreturn__ = gdImageCreateTrueColor( sx, sy );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_GifAnimBeginPtr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int size = (int) vm->param(1)->forceInteger();
int GlobalCM = (int) vm->param(2)->forceInteger();
int Loops = (int) vm->param(3)->forceInteger();

   void* __funcreturn__ = gdImageGifAnimBeginPtr( im,    &size, GlobalCM, Loops );
   *vm->param(1) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_Char( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdFont* f = (gdFont*) vm->param(1)->asObject()->getUserData();
int x = (int) vm->param(2)->forceInteger();
int y = (int) vm->param(3)->forceInteger();
int c = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageChar( im, f, x, y, c, color );
}


static void _falbind_GdImage_BoundsSafe( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x = (int) vm->param(1)->forceInteger();
int y = (int) vm->param(2)->forceInteger();

   int __funcreturn__ = gdImageBoundsSafe( im, x, y );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_StringFT( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isString()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isString()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N,S,N,N,N,N,S" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int brect = (int) vm->param(1)->forceInteger();
int fg = (int) vm->param(2)->forceInteger();
   AutoCString autoc_fontlist( *vm->param(3)->asString() );
   char const* fontlist = (char const*) autoc_fontlist.c_str();
double ptsize = (double) vm->param(4)->forceNumeric();
double angle = (double) vm->param(5)->forceNumeric();
int x = (int) vm->param(6)->forceInteger();
int y = (int) vm->param(7)->forceInteger();
   AutoCString autoc_string( *vm->param(8)->asString() );
   char const* string = (char const*) autoc_string.c_str();

   char* __funcreturn__ = gdImageStringFT( im,    &brect, fg, fontlist, ptsize, angle, x, y, string );
   *vm->param(1) = (int64) brect;
   vm->retval( new CoreString( __funcreturn__, -1 ));
}


static void _falbind_GdImage_SetStyle( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int style = (int) vm->param(1)->forceInteger();
int noOfPixels = (int) vm->param(2)->forceInteger();

   gdImageSetStyle( im,    &style, noOfPixels );
   *vm->param(1) = (int64) style;
}


static void _falbind_GdImage_PngPtr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int size = (int) vm->param(1)->forceInteger();

   void* __funcreturn__ = gdImagePngPtr( im,    &size );
   *vm->param(1) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_Sharpen( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int pct = (int) vm->param(1)->forceInteger();

   gdImageSharpen( im, pct );
}


static void _falbind_gdImageCreateFromPngPtr( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isMemBuf()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,M" ) );
   }

int size = (int) vm->param(0)->forceInteger();
void* data = (void*) vm->param(1)->asMemBuf()->data();

   gdImageStruct* __funcreturn__ = gdImageCreateFromPngPtr( size, data );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_Polygon( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdPoint* p = (gdPoint*) vm->param(1)->asObject()->getUserData();
int n = (int) vm->param(2)->forceInteger();
int c = (int) vm->param(3)->forceInteger();

   gdImagePolygon( im, p, n, c );
}


static void _falbind_gdImageCreateFromGd2Ptr( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isMemBuf()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,M" ) );
   }

int size = (int) vm->param(0)->forceInteger();
void* data = (void*) vm->param(1)->asMemBuf()->data();

   gdImageStruct* __funcreturn__ = gdImageCreateFromGd2Ptr( size, data );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_Ellipse( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int cx = (int) vm->param(1)->forceInteger();
int cy = (int) vm->param(2)->forceInteger();
int w = (int) vm->param(3)->forceInteger();
int h = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageEllipse( im, cx, cy, w, h, color );
}


static void _falbind_GdImage_Ellipse( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int cx = (int) vm->param(1)->forceInteger();
int cy = (int) vm->param(2)->forceInteger();
int w = (int) vm->param(3)->forceInteger();
int h = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageEllipse( im, cx, cy, w, h, color );
}


static void _falbind_GdImage_SetTile( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* tile = (gdImage*) vm->param(1)->asObject()->getUserData();

   gdImageSetTile( im, tile );
}


static void _falbind_GdImage_GetTrueColorPixel( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x = (int) vm->param(1)->forceInteger();
int y = (int) vm->param(2)->forceInteger();

   int __funcreturn__ = gdImageGetTrueColorPixel( im, x, y );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_ColorClosest( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();

   int __funcreturn__ = gdImageColorClosest( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_Copy( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage,N,N,N,N,N,N" ) );
   }

gdImage* dst = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* src = (gdImage*) vm->param(1)->asObject()->getUserData();
int dstX = (int) vm->param(2)->forceInteger();
int dstY = (int) vm->param(3)->forceInteger();
int srcX = (int) vm->param(4)->forceInteger();
int srcY = (int) vm->param(5)->forceInteger();
int w = (int) vm->param(6)->forceInteger();
int h = (int) vm->param(7)->forceInteger();

   gdImageCopy( dst, src, dstX, dstY, srcX, srcY, w, h );
}


static void _falbind_GdImage_String16( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !((vm->param(4)->isOrdinal() && vm->isParamByRef(4))))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N,$N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdFont* f = (gdFont*) vm->param(1)->asObject()->getUserData();
int x = (int) vm->param(2)->forceInteger();
int y = (int) vm->param(3)->forceInteger();
unsigned short s = (unsigned short) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageString16( im, f, x, y,    &s, color );
   *vm->param(4) = (int64) s;
}


static void _falbind_GdImage_SaveAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int saveAlphaArg = (int) vm->param(1)->forceInteger();

   gdImageSaveAlpha( im, saveAlphaArg );
}


static void _falbind_GdImage_OpenPolygon( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdPoint* p = (gdPoint*) vm->param(1)->asObject()->getUserData();
int n = (int) vm->param(2)->forceInteger();
int c = (int) vm->param(3)->forceInteger();

   gdImageOpenPolygon( im, p, n, c );
}


static void _falbind_GdImage_SetThickness( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int thickness = (int) vm->param(1)->forceInteger();

   gdImageSetThickness( im, thickness );
}


static void _falbind_GdImage_GifPtr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int size = (int) vm->param(1)->forceInteger();

   void* __funcreturn__ = gdImageGifPtr( im,    &size );
   *vm->param(1) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_StringTTF( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isString()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isString()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N,S,N,N,N,N,S" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int brect = (int) vm->param(1)->forceInteger();
int fg = (int) vm->param(2)->forceInteger();
   AutoCString autoc_fontlist( *vm->param(3)->asString() );
   char const* fontlist = (char const*) autoc_fontlist.c_str();
double ptsize = (double) vm->param(4)->forceNumeric();
double angle = (double) vm->param(5)->forceNumeric();
int x = (int) vm->param(6)->forceInteger();
int y = (int) vm->param(7)->forceInteger();
   AutoCString autoc_string( *vm->param(8)->asString() );
   char const* string = (char const*) autoc_string.c_str();

   char* __funcreturn__ = gdImageStringTTF( im,    &brect, fg, fontlist, ptsize, angle, x, y, string );
   *vm->param(1) = (int64) brect;
   vm->retval( new CoreString( __funcreturn__, -1 ));
}


static void _falbind_GdImage_ColorAllocateAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();
int a = (int) vm->param(4)->forceInteger();

   int __funcreturn__ = gdImageColorAllocateAlpha( im, r, g, b, a );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_GifAnimEndPtr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !((vm->param(0)->isOrdinal() && vm->isParamByRef(0))))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "$N" ) );
   }

int size = (int) vm->param(0)->forceInteger();

   void* __funcreturn__ = gdImageGifAnimEndPtr(    &size );
   *vm->param(0) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_AABlend( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();

   gdImageAABlend( im );
}


static void _falbind_GdImage_Interlace( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int interlaceArg = (int) vm->param(1)->forceInteger();

   gdImageInterlace( im, interlaceArg );
}


static void _falbind_GdImage_StringFTCircle( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isString()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isString()))
        ||( vm->param(9) == 0 || !(   vm->param(9)->isString()))
        ||( vm->param(10) == 0 || !(   vm->param(10)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N,S,N,S,S,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int cx = (int) vm->param(1)->forceInteger();
int cy = (int) vm->param(2)->forceInteger();
double radius = (double) vm->param(3)->forceNumeric();
double textRadius = (double) vm->param(4)->forceNumeric();
double fillPortion = (double) vm->param(5)->forceNumeric();
   AutoCString autoc_font( *vm->param(6)->asString() );
   char const* font = (char const*) autoc_font.c_str();
double points = (double) vm->param(7)->forceNumeric();
   AutoCString autoc_top( *vm->param(8)->asString() );
   char const* top = (char const*) autoc_top.c_str();
   AutoCString autoc_bottom( *vm->param(9)->asString() );
   char const* bottom = (char const*) autoc_bottom.c_str();
int fgcolor = (int) vm->param(10)->forceInteger();

   char* __funcreturn__ = gdImageStringFTCircle( im, cx, cy, radius, textRadius, fillPortion, font, points, top, bottom, fgcolor );
   vm->retval( new CoreString( __funcreturn__, -1 ));
}


static void _falbind_GdImage_SetAntiAliased( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int c = (int) vm->param(1)->forceInteger();

   gdImageSetAntiAliased( im, c );
}


static void _falbind_GdImage_FilledEllipse( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int cx = (int) vm->param(1)->forceInteger();
int cy = (int) vm->param(2)->forceInteger();
int w = (int) vm->param(3)->forceInteger();
int h = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageFilledEllipse( im, cx, cy, w, h, color );
}


static void _falbind_GdImage_SetBrush( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* brush = (gdImage*) vm->param(1)->asObject()->getUserData();

   gdImageSetBrush( im, brush );
}


static void _falbind_GdImage_CopyMerge( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage,N,N,N,N,N,N,N" ) );
   }

gdImage* dst = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* src = (gdImage*) vm->param(1)->asObject()->getUserData();
int dstX = (int) vm->param(2)->forceInteger();
int dstY = (int) vm->param(3)->forceInteger();
int srcX = (int) vm->param(4)->forceInteger();
int srcY = (int) vm->param(5)->forceInteger();
int w = (int) vm->param(6)->forceInteger();
int h = (int) vm->param(7)->forceInteger();
int pct = (int) vm->param(8)->forceInteger();

   gdImageCopyMerge( dst, src, dstX, dstY, srcX, srcY, w, h, pct );
}


static void _falbind_GdImage_FillToBorder( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x = (int) vm->param(1)->forceInteger();
int y = (int) vm->param(2)->forceInteger();
int border = (int) vm->param(3)->forceInteger();
int color = (int) vm->param(4)->forceInteger();

   gdImageFillToBorder( im, x, y, border, color );
}


static void _falbind_GdImage_JpegPtr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int size = (int) vm->param(1)->forceInteger();
int quality = (int) vm->param(2)->forceInteger();

   void* __funcreturn__ = gdImageJpegPtr( im,    &size, quality );
   *vm->param(1) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_CopyResampled( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isOrdinal()))
        ||( vm->param(9) == 0 || !(   vm->param(9)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage,N,N,N,N,N,N,N,N" ) );
   }

gdImage* dst = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* src = (gdImage*) vm->param(1)->asObject()->getUserData();
int dstX = (int) vm->param(2)->forceInteger();
int dstY = (int) vm->param(3)->forceInteger();
int srcX = (int) vm->param(4)->forceInteger();
int srcY = (int) vm->param(5)->forceInteger();
int dstW = (int) vm->param(6)->forceInteger();
int dstH = (int) vm->param(7)->forceInteger();
int srcW = (int) vm->param(8)->forceInteger();
int srcH = (int) vm->param(9)->forceInteger();

   gdImageCopyResampled( dst, src, dstX, dstY, srcX, srcY, dstW, dstH, srcW, srcH );
}


static void _falbind_GdImage_GetClip( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !((vm->param(2)->isOrdinal() && vm->isParamByRef(2))))
        ||( vm->param(3) == 0 || !((vm->param(3)->isOrdinal() && vm->isParamByRef(3))))
        ||( vm->param(4) == 0 || !((vm->param(4)->isOrdinal() && vm->isParamByRef(4))))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,$N,$N,$N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x1P = (int) vm->param(1)->forceInteger();
int y1P = (int) vm->param(2)->forceInteger();
int x2P = (int) vm->param(3)->forceInteger();
int y2P = (int) vm->param(4)->forceInteger();

   gdImageGetClip( im,    &x1P,    &y1P,    &x2P,    &y2P );
   *vm->param(1) = (int64) x1P;
   *vm->param(2) = (int64) y1P;
   *vm->param(3) = (int64) x2P;
   *vm->param(4) = (int64) y2P;
}


static void _falbind_GdImage_Line( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x1 = (int) vm->param(1)->forceInteger();
int y1 = (int) vm->param(2)->forceInteger();
int x2 = (int) vm->param(3)->forceInteger();
int y2 = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageLine( im, x1, y1, x2, y2, color );
}


static void _falbind_GdImage_PngPtrEx( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int size = (int) vm->param(1)->forceInteger();
int level = (int) vm->param(2)->forceInteger();

   void* __funcreturn__ = gdImagePngPtrEx( im,    &size, level );
   *vm->param(1) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_Compare( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage" ) );
   }

gdImage* im1 = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* im2 = (gdImage*) vm->param(1)->asObject()->getUserData();

   int __funcreturn__ = gdImageCompare( im1, im2 );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_gdImageCreateFromGdPtr( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isMemBuf()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,M" ) );
   }

int size = (int) vm->param(0)->forceInteger();
void* data = (void*) vm->param(1)->asMemBuf()->data();

   gdImageStruct* __funcreturn__ = gdImageCreateFromGdPtr( size, data );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_FilledPolygon( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdPoint* p = (gdPoint*) vm->param(1)->asObject()->getUserData();
int n = (int) vm->param(2)->forceInteger();
int c = (int) vm->param(3)->forceInteger();

   gdImageFilledPolygon( im, p, n, c );
}


static void _falbind_GdImage_SquareToCircle( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int radius = (int) vm->param(1)->forceInteger();

   gdImagePtr __funcreturn__ = gdImageSquareToCircle( im, radius );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_ColorResolve( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();

   int __funcreturn__ = gdImageColorResolve( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_ColorTransparent( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int color = (int) vm->param(1)->forceInteger();

   gdImageColorTransparent( im, color );
}


static void _falbind_GdImage_PaletteCopy( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage" ) );
   }

gdImage* dst = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* src = (gdImage*) vm->param(1)->asObject()->getUserData();

   gdImagePaletteCopy( dst, src );
}


static void _falbind_gdImageCreateFromGd2PartPtr( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isMemBuf()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,M,N,N,N,N" ) );
   }

int size = (int) vm->param(0)->forceInteger();
void* data = (void*) vm->param(1)->asMemBuf()->data();
int srcx = (int) vm->param(2)->forceInteger();
int srcy = (int) vm->param(3)->forceInteger();
int w = (int) vm->param(4)->forceInteger();
int h = (int) vm->param(5)->forceInteger();

   gdImageStruct* __funcreturn__ = gdImageCreateFromGd2PartPtr( size, data, srcx, srcy, w, h );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_ColorClosestAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();
int a = (int) vm->param(4)->forceInteger();

   int __funcreturn__ = gdImageColorClosestAlpha( im, r, g, b, a );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_TrueColorToPalette( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int ditherFlag = (int) vm->param(1)->forceInteger();
int colorsWanted = (int) vm->param(2)->forceInteger();

   gdImageTrueColorToPalette( im, ditherFlag, colorsWanted );
}


static void _falbind_gdImage_init( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

int sx = (int) vm->param(0)->forceInteger();
int sy = (int) vm->param(1)->forceInteger();

   gdImageStruct* __funcreturn__ = gdImageCreate( sx, sy );
   vm->self().asObject()->setUserData( __funcreturn__ );
}


static void _falbind_GdImage_StringFTEx( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isString()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isString()))
        ||( vm->param(9) == 0 || !(vm->param(9)->isObject() && vm->param(9)->asObjectSafe()->derivedFrom("gdFTStringExtra$")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N,S,N,N,N,N,S,gdFTStringExtra$" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int brect = (int) vm->param(1)->forceInteger();
int fg = (int) vm->param(2)->forceInteger();
   AutoCString autoc_fontlist( *vm->param(3)->asString() );
   char const* fontlist = (char const*) autoc_fontlist.c_str();
double ptsize = (double) vm->param(4)->forceNumeric();
double angle = (double) vm->param(5)->forceNumeric();
int x = (int) vm->param(6)->forceInteger();
int y = (int) vm->param(7)->forceInteger();
   AutoCString autoc_string( *vm->param(8)->asString() );
   char const* string = (char const*) autoc_string.c_str();
gdFTStringExtra* strex = (gdFTStringExtra*) vm->param(9)->asObject()->getUserData();

   char* __funcreturn__ = gdImageStringFTEx( im,    &brect, fg, fontlist, ptsize, angle, x, y, string, strex );
   *vm->param(1) = (int64) brect;
   vm->retval( new CoreString( __funcreturn__, -1 ));
}


static void _falbind_GdImage_Fill( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x = (int) vm->param(1)->forceInteger();
int y = (int) vm->param(2)->forceInteger();
int color = (int) vm->param(3)->forceInteger();

   gdImageFill( im, x, y, color );
}


static void _falbind_GdImage_CopyMergeGray( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage,N,N,N,N,N,N,N" ) );
   }

gdImage* dst = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* src = (gdImage*) vm->param(1)->asObject()->getUserData();
int dstX = (int) vm->param(2)->forceInteger();
int dstY = (int) vm->param(3)->forceInteger();
int srcX = (int) vm->param(4)->forceInteger();
int srcY = (int) vm->param(5)->forceInteger();
int w = (int) vm->param(6)->forceInteger();
int h = (int) vm->param(7)->forceInteger();
int pct = (int) vm->param(8)->forceInteger();

   gdImageCopyMergeGray( dst, src, dstX, dstY, srcX, srcY, w, h, pct );
}


static void _falbind_GdImage_ColorClosestHWB( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();

   int __funcreturn__ = gdImageColorClosestHWB( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_ColorDeallocate( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int color = (int) vm->param(1)->forceInteger();

   gdImageColorDeallocate( im, color );
}


static void _falbind_GdImage_CopyResized( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isOrdinal()))
        ||( vm->param(9) == 0 || !(   vm->param(9)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage,N,N,N,N,N,N,N,N" ) );
   }

gdImage* dst = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* src = (gdImage*) vm->param(1)->asObject()->getUserData();
int dstX = (int) vm->param(2)->forceInteger();
int dstY = (int) vm->param(3)->forceInteger();
int srcX = (int) vm->param(4)->forceInteger();
int srcY = (int) vm->param(5)->forceInteger();
int dstW = (int) vm->param(6)->forceInteger();
int dstH = (int) vm->param(7)->forceInteger();
int srcW = (int) vm->param(8)->forceInteger();
int srcH = (int) vm->param(9)->forceInteger();

   gdImageCopyResized( dst, src, dstX, dstY, srcX, srcY, dstW, dstH, srcW, srcH );
}


static void _falbind_GdImage_AlphaBlending( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int alphaBlendingArg = (int) vm->param(1)->forceInteger();

   gdImageAlphaBlending( im, alphaBlendingArg );
}


static void _falbind_gdImageCreateFromGifPtr( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isMemBuf()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,M" ) );
   }

int size = (int) vm->param(0)->forceInteger();
void* data = (void*) vm->param(1)->asMemBuf()->data();

   gdImageStruct* __funcreturn__ = gdImageCreateFromGifPtr( size, data );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_CharUp( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdFont* f = (gdFont*) vm->param(1)->asObject()->getUserData();
int x = (int) vm->param(2)->forceInteger();
int y = (int) vm->param(3)->forceInteger();
int c = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageCharUp( im, f, x, y, c, color );
}


static void _falbind_GdImage_SetClip( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x1 = (int) vm->param(1)->forceInteger();
int y1 = (int) vm->param(2)->forceInteger();
int x2 = (int) vm->param(3)->forceInteger();
int y2 = (int) vm->param(4)->forceInteger();

   gdImageSetClip( im, x1, y1, x2, y2 );
}


static void _falbind_gdImageCreateFromWBMPPtr( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isMemBuf()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,M" ) );
   }

int size = (int) vm->param(0)->forceInteger();
void* data = (void*) vm->param(1)->asMemBuf()->data();

   gdImageStruct* __funcreturn__ = gdImageCreateFromWBMPPtr( size, data );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_StringUp16( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !((vm->param(4)->isOrdinal() && vm->isParamByRef(4))))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N,$N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdFont* f = (gdFont*) vm->param(1)->asObject()->getUserData();
int x = (int) vm->param(2)->forceInteger();
int y = (int) vm->param(3)->forceInteger();
unsigned short s = (unsigned short) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageStringUp16( im, f, x, y,    &s, color );
   *vm->param(4) = (int64) s;
}


static void _falbind_GdImage_SetPixel( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x = (int) vm->param(1)->forceInteger();
int y = (int) vm->param(2)->forceInteger();
int color = (int) vm->param(3)->forceInteger();

   gdImageSetPixel( im, x, y, color );
}


static void _falbind_GdImage_String( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isString()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N,S,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdFont* f = (gdFont*) vm->param(1)->asObject()->getUserData();
int x = (int) vm->param(2)->forceInteger();
int y = (int) vm->param(3)->forceInteger();
   AutoCString autoc_s( *vm->param(4)->asString() );
   char const* s = (char const*) autoc_s.c_str();
int color = (int) vm->param(5)->forceInteger();

   gdImageString( im, f, x, y, s, color );
}


static void _falbind_GdImage_FilledArc( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int cx = (int) vm->param(1)->forceInteger();
int cy = (int) vm->param(2)->forceInteger();
int w = (int) vm->param(3)->forceInteger();
int h = (int) vm->param(4)->forceInteger();
int s = (int) vm->param(5)->forceInteger();
int e = (int) vm->param(6)->forceInteger();
int color = (int) vm->param(7)->forceInteger();
int style = (int) vm->param(8)->forceInteger();

   gdImageFilledArc( im, cx, cy, w, h, s, e, color, style );
}


static void _falbind_GdImage_WBMPPtr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int size = (int) vm->param(1)->forceInteger();
int fg = (int) vm->param(2)->forceInteger();

   void* __funcreturn__ = gdImageWBMPPtr( im,    &size, fg );
   *vm->param(1) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_ColorExact( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();

   int __funcreturn__ = gdImageColorExact( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_Gd2Ptr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !((vm->param(3)->isOrdinal() && vm->isParamByRef(3))))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,$N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int cs = (int) vm->param(1)->forceInteger();
int fmt = (int) vm->param(2)->forceInteger();
int size = (int) vm->param(3)->forceInteger();

   void* __funcreturn__ = gdImageGd2Ptr( im, cs, fmt,    &size );
   *vm->param(3) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_SetAntiAliasedDontBlend( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int c = (int) vm->param(1)->forceInteger();
int dont_blend = (int) vm->param(2)->forceInteger();

   gdImageSetAntiAliasedDontBlend( im, c, dont_blend );
}


static void _falbind_GdImage_StringUp( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isString()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdFont,N,N,S,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
gdFont* f = (gdFont*) vm->param(1)->asObject()->getUserData();
int x = (int) vm->param(2)->forceInteger();
int y = (int) vm->param(3)->forceInteger();
   AutoCString autoc_s( *vm->param(4)->asString() );
   char const* s = (char const*) autoc_s.c_str();
int color = (int) vm->param(5)->forceInteger();

   gdImageStringUp( im, f, x, y, s, color );
}


static void _falbind_GdImage_CreatePaletteFromTrueColor( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int ditherFlag = (int) vm->param(1)->forceInteger();
int colorsWanted = (int) vm->param(2)->forceInteger();

   gdImageStruct* __funcreturn__ = gdImageCreatePaletteFromTrueColor( im, ditherFlag, colorsWanted );
CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_GetPixel( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x = (int) vm->param(1)->forceInteger();
int y = (int) vm->param(2)->forceInteger();

   int __funcreturn__ = gdImageGetPixel( im, x, y );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_ColorExactAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();
int a = (int) vm->param(4)->forceInteger();

   int __funcreturn__ = gdImageColorExactAlpha( im, r, g, b, a );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_GifAnimAddPtr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(vm->param(7)->isObject() && vm->param(7)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N,N,N,N,N,N,GdImage" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int size = (int) vm->param(1)->forceInteger();
int LocalCM = (int) vm->param(2)->forceInteger();
int LeftOfs = (int) vm->param(3)->forceInteger();
int TopOfs = (int) vm->param(4)->forceInteger();
int Delay = (int) vm->param(5)->forceInteger();
int Disposal = (int) vm->param(6)->forceInteger();
gdImage* previm = (gdImage*) vm->param(7)->asObject()->getUserData();

   void* __funcreturn__ = gdImageGifAnimAddPtr( im,    &size, LocalCM, LeftOfs, TopOfs, Delay, Disposal, previm );
   *vm->param(1) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_Rectangle( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x1 = (int) vm->param(1)->forceInteger();
int y1 = (int) vm->param(2)->forceInteger();
int x2 = (int) vm->param(3)->forceInteger();
int y2 = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageRectangle( im, x1, y1, x2, y2, color );
}


static void _falbind_GdImage_GdPtr( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !((vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,$N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int size = (int) vm->param(1)->forceInteger();

   void* __funcreturn__ = gdImageGdPtr( im,    &size );
   *vm->param(1) = (int64) size;
   vm->retval( new MemBuf_1( __funcreturn__, 0xFFFFFFFF, false );
}


static void _falbind_GdImage_Arc( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int cx = (int) vm->param(1)->forceInteger();
int cy = (int) vm->param(2)->forceInteger();
int w = (int) vm->param(3)->forceInteger();
int h = (int) vm->param(4)->forceInteger();
int s = (int) vm->param(5)->forceInteger();
int e = (int) vm->param(6)->forceInteger();
int color = (int) vm->param(7)->forceInteger();

   gdImageArc( im, cx, cy, w, h, s, e, color );
}


static void _falbind_GdImage_ColorResolveAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();
int a = (int) vm->param(4)->forceInteger();

   int __funcreturn__ = gdImageColorResolveAlpha( im, r, g, b, a );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_ColorAllocate( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int r = (int) vm->param(1)->forceInteger();
int g = (int) vm->param(2)->forceInteger();
int b = (int) vm->param(3)->forceInteger();

   int __funcreturn__ = gdImageColorAllocate( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_DashedLine( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x1 = (int) vm->param(1)->forceInteger();
int y1 = (int) vm->param(2)->forceInteger();
int x2 = (int) vm->param(3)->forceInteger();
int y2 = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageDashedLine( im, x1, y1, x2, y2, color );
}


static void _falbind_GdImage_FilledRectangle( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
int x1 = (int) vm->param(1)->forceInteger();
int y1 = (int) vm->param(2)->forceInteger();
int x2 = (int) vm->param(3)->forceInteger();
int y2 = (int) vm->param(4)->forceInteger();
int color = (int) vm->param(5)->forceInteger();

   gdImageFilledRectangle( im, x1, y1, x2, y2, color );
}


static void _falbind_GdImage_CopyRotated( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(vm->param(1)->isObject() && vm->param(1)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
        ||( vm->param(8) == 0 || !(   vm->param(8)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,GdImage,N,N,N,N,N,N,N" ) );
   }

gdImage* dst = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImage* src = (gdImage*) vm->param(1)->asObject()->getUserData();
double dstX = (double) vm->param(2)->forceNumeric();
double dstY = (double) vm->param(3)->forceNumeric();
int srcX = (int) vm->param(4)->forceInteger();
int srcY = (int) vm->param(5)->forceInteger();
int srcWidth = (int) vm->param(6)->forceInteger();
int srcHeight = (int) vm->param(7)->forceInteger();
int angle = (int) vm->param(8)->forceInteger();

   gdImageCopyRotated( dst, src, dstX, dstY, srcX, srcY, srcWidth, srcHeight, angle );
}


FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self
   Falcon::Module *self = new Falcon::Module();
   
   self->name( "gd2" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( 1, 0, 0 );
   Symbol *sym_GdFont = self->addClass("GdFont");
   sym_GdFont->setWKS( true );
   sym_GdFont->getClassDef()->factory( &_falbind_GdFont_factory );

   Symbol *sym_GdFont = self->addClass("GdFont");
   sym_GdFont->setWKS( true );
   sym_GdFont->getClassDef()->factory( &_falbind_GdFont_factory );

   Symbol *sym_gdFTStringExtra$ = self->addClass("gdFTStringExtra$");
   sym_gdFTStringExtra$->setWKS( true );
   sym_gdFTStringExtra$->getClassDef()->factory( &_falbind_gdFTStringExtra$_factory );

   Symbol *sym_GdImage = self->addClass("GdImage");
   sym_GdImage->setWKS( true );
   sym_GdImage->getClassDef()->factory( &_falbind_GdImage_factory );

   self->addExtFunc( "gdImageCreateFrom(Jpeg)Ptr", _falbind_gdImageCreateFromJpegPtr )
      ->addParam( "size" )->addParam( "data" );
   self->addClassMethod( sym_GdImage, "CreateTrueColor", &_falbind_GdImage_CreateTrueColor ).asSymbol()
      ->addParam( "sx" )->addParam( "sy" );
   self->addClassMethod( sym_GdImage, "GifAnimBeginPtr", &_falbind_GdImage_GifAnimBeginPtr ).asSymbol()
      ->addParam( "im" )->addParam( "size" )->addParam( "GlobalCM" )->addParam( "Loops" );
   self->addClassMethod( sym_GdImage, "Char", &_falbind_GdImage_Char ).asSymbol()
      ->addParam( "im" )->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "c" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "BoundsSafe", &_falbind_GdImage_BoundsSafe ).asSymbol()
      ->addParam( "im" )->addParam( "x" )->addParam( "y" );
   self->addClassMethod( sym_GdImage, "StringFT", &_falbind_GdImage_StringFT ).asSymbol()
      ->addParam( "im" )->addParam( "brect" )->addParam( "fg" )->addParam( "fontlist" )->addParam( "ptsize" )->addParam( "angle" )->addParam( "x" )->addParam( "y" )->addParam( "string" );
   self->addClassMethod( sym_GdImage, "SetStyle", &_falbind_GdImage_SetStyle ).asSymbol()
      ->addParam( "im" )->addParam( "style" )->addParam( "noOfPixels" );
   self->addClassMethod( sym_GdImage, "PngPtr", &_falbind_GdImage_PngPtr ).asSymbol()
      ->addParam( "im" )->addParam( "size" );
   self->addClassMethod( sym_GdImage, "Sharpen", &_falbind_GdImage_Sharpen ).asSymbol()
      ->addParam( "im" )->addParam( "pct" );
   self->addExtFunc( "gdImageCreateFrom(Png)Ptr", _falbind_gdImageCreateFromPngPtr )
      ->addParam( "size" )->addParam( "data" );
   self->addClassMethod( sym_GdImage, "Polygon", &_falbind_GdImage_Polygon ).asSymbol()
      ->addParam( "im" )->addParam( "p" )->addParam( "n" )->addParam( "c" );
   self->addExtFunc( "gdImageCreateFrom(Gd2)Ptr", _falbind_gdImageCreateFromGd2Ptr )
      ->addParam( "size" )->addParam( "data" );
   self->addClassMethod( sym_GdImage, "Ellipse", &_falbind_GdImage_Ellipse ).asSymbol()
      ->addParam( "im" )->addParam( "cx" )->addParam( "cy" )->addParam( "w" )->addParam( "h" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "Ellipse", &_falbind_GdImage_Ellipse ).asSymbol()
      ->addParam( "im" )->addParam( "cx" )->addParam( "cy" )->addParam( "w" )->addParam( "h" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "SetTile", &_falbind_GdImage_SetTile ).asSymbol()
      ->addParam( "im" )->addParam( "tile" );
   self->addClassMethod( sym_GdImage, "GetTrueColorPixel", &_falbind_GdImage_GetTrueColorPixel ).asSymbol()
      ->addParam( "im" )->addParam( "x" )->addParam( "y" );
   self->addClassMethod( sym_GdImage, "ColorClosest", &_falbind_GdImage_ColorClosest ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "Copy", &_falbind_GdImage_Copy ).asSymbol()
      ->addParam( "dst" )->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "w" )->addParam( "h" );
   self->addClassMethod( sym_GdImage, "String16", &_falbind_GdImage_String16 ).asSymbol()
      ->addParam( "im" )->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "s" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "SaveAlpha", &_falbind_GdImage_SaveAlpha ).asSymbol()
      ->addParam( "im" )->addParam( "saveAlphaArg" );
   self->addClassMethod( sym_GdImage, "OpenPolygon", &_falbind_GdImage_OpenPolygon ).asSymbol()
      ->addParam( "im" )->addParam( "p" )->addParam( "n" )->addParam( "c" );
   self->addClassMethod( sym_GdImage, "SetThickness", &_falbind_GdImage_SetThickness ).asSymbol()
      ->addParam( "im" )->addParam( "thickness" );
   self->addClassMethod( sym_GdImage, "GifPtr", &_falbind_GdImage_GifPtr ).asSymbol()
      ->addParam( "im" )->addParam( "size" );
   self->addClassMethod( sym_GdImage, "StringTTF", &_falbind_GdImage_StringTTF ).asSymbol()
      ->addParam( "im" )->addParam( "brect" )->addParam( "fg" )->addParam( "fontlist" )->addParam( "ptsize" )->addParam( "angle" )->addParam( "x" )->addParam( "y" )->addParam( "string" );
   self->addClassMethod( sym_GdImage, "ColorAllocateAlpha", &_falbind_GdImage_ColorAllocateAlpha ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" )->addParam( "a" );
   self->addClassMethod( sym_GdImage, "GifAnimEndPtr", &_falbind_GdImage_GifAnimEndPtr ).asSymbol()
      ->addParam( "size" );
   self->addClassMethod( sym_GdImage, "AABlend", &_falbind_GdImage_AABlend ).asSymbol()
      ->addParam( "im" );
   self->addClassMethod( sym_GdImage, "Interlace", &_falbind_GdImage_Interlace ).asSymbol()
      ->addParam( "im" )->addParam( "interlaceArg" );
   self->addClassMethod( sym_GdImage, "StringFTCircle", &_falbind_GdImage_StringFTCircle ).asSymbol()
      ->addParam( "im" )->addParam( "cx" )->addParam( "cy" )->addParam( "radius" )->addParam( "textRadius" )->addParam( "fillPortion" )->addParam( "font" )->addParam( "points" )->addParam( "top" )->addParam( "bottom" )->addParam( "fgcolor" );
   self->addClassMethod( sym_GdImage, "SetAntiAliased", &_falbind_GdImage_SetAntiAliased ).asSymbol()
      ->addParam( "im" )->addParam( "c" );
   self->addClassMethod( sym_GdImage, "FilledEllipse", &_falbind_GdImage_FilledEllipse ).asSymbol()
      ->addParam( "im" )->addParam( "cx" )->addParam( "cy" )->addParam( "w" )->addParam( "h" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "SetBrush", &_falbind_GdImage_SetBrush ).asSymbol()
      ->addParam( "im" )->addParam( "brush" );
   self->addClassMethod( sym_GdImage, "CopyMerge", &_falbind_GdImage_CopyMerge ).asSymbol()
      ->addParam( "dst" )->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "w" )->addParam( "h" )->addParam( "pct" );
   self->addClassMethod( sym_GdImage, "FillToBorder", &_falbind_GdImage_FillToBorder ).asSymbol()
      ->addParam( "im" )->addParam( "x" )->addParam( "y" )->addParam( "border" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "JpegPtr", &_falbind_GdImage_JpegPtr ).asSymbol()
      ->addParam( "im" )->addParam( "size" )->addParam( "quality" );
   self->addClassMethod( sym_GdImage, "CopyResampled", &_falbind_GdImage_CopyResampled ).asSymbol()
      ->addParam( "dst" )->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "dstW" )->addParam( "dstH" )->addParam( "srcW" )->addParam( "srcH" );
   self->addClassMethod( sym_GdImage, "GetClip", &_falbind_GdImage_GetClip ).asSymbol()
      ->addParam( "im" )->addParam( "x1P" )->addParam( "y1P" )->addParam( "x2P" )->addParam( "y2P" );
   self->addClassMethod( sym_GdImage, "Line", &_falbind_GdImage_Line ).asSymbol()
      ->addParam( "im" )->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "PngPtrEx", &_falbind_GdImage_PngPtrEx ).asSymbol()
      ->addParam( "im" )->addParam( "size" )->addParam( "level" );
   self->addClassMethod( sym_GdImage, "Compare", &_falbind_GdImage_Compare ).asSymbol()
      ->addParam( "im1" )->addParam( "im2" );
   self->addExtFunc( "gdImageCreateFrom(Gd)Ptr", _falbind_gdImageCreateFromGdPtr )
      ->addParam( "size" )->addParam( "data" );
   self->addClassMethod( sym_GdImage, "FilledPolygon", &_falbind_GdImage_FilledPolygon ).asSymbol()
      ->addParam( "im" )->addParam( "p" )->addParam( "n" )->addParam( "c" );
   self->addClassMethod( sym_GdImage, "SquareToCircle", &_falbind_GdImage_SquareToCircle ).asSymbol()
      ->addParam( "im" )->addParam( "radius" );
   self->addClassMethod( sym_GdImage, "ColorResolve", &_falbind_GdImage_ColorResolve ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "ColorTransparent", &_falbind_GdImage_ColorTransparent ).asSymbol()
      ->addParam( "im" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "PaletteCopy", &_falbind_GdImage_PaletteCopy ).asSymbol()
      ->addParam( "dst" )->addParam( "src" );
   self->addExtFunc( "gdImageCreateFrom(Gd2Part)Ptr", _falbind_gdImageCreateFromGd2PartPtr )
      ->addParam( "size" )->addParam( "data" )->addParam( "srcx" )->addParam( "srcy" )->addParam( "w" )->addParam( "h" );
   self->addClassMethod( sym_GdImage, "ColorClosestAlpha", &_falbind_GdImage_ColorClosestAlpha ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" )->addParam( "a" );
   self->addClassMethod( sym_GdImage, "TrueColorToPalette", &_falbind_GdImage_TrueColorToPalette ).asSymbol()
      ->addParam( "im" )->addParam( "ditherFlag" )->addParam( "colorsWanted" );
   Symbol* sym_gdImage_init = self->addExtFunc( "init", &_falbind_gdImage_init, false )
      ->addParam( "sx" )->addParam( "sy" );
   sym_gdImage->getClassDef()->constructor( sym_gdImage_init );
   self->addClassMethod( sym_GdImage, "StringFTEx", &_falbind_GdImage_StringFTEx ).asSymbol()
      ->addParam( "im" )->addParam( "brect" )->addParam( "fg" )->addParam( "fontlist" )->addParam( "ptsize" )->addParam( "angle" )->addParam( "x" )->addParam( "y" )->addParam( "string" )->addParam( "strex" );
   self->addClassMethod( sym_GdImage, "Fill", &_falbind_GdImage_Fill ).asSymbol()
      ->addParam( "im" )->addParam( "x" )->addParam( "y" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "CopyMergeGray", &_falbind_GdImage_CopyMergeGray ).asSymbol()
      ->addParam( "dst" )->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "w" )->addParam( "h" )->addParam( "pct" );
   self->addClassMethod( sym_GdImage, "ColorClosestHWB", &_falbind_GdImage_ColorClosestHWB ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "ColorDeallocate", &_falbind_GdImage_ColorDeallocate ).asSymbol()
      ->addParam( "im" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "CopyResized", &_falbind_GdImage_CopyResized ).asSymbol()
      ->addParam( "dst" )->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "dstW" )->addParam( "dstH" )->addParam( "srcW" )->addParam( "srcH" );
   self->addClassMethod( sym_GdImage, "AlphaBlending", &_falbind_GdImage_AlphaBlending ).asSymbol()
      ->addParam( "im" )->addParam( "alphaBlendingArg" );
   self->addExtFunc( "gdImageCreateFrom(Gif)Ptr", _falbind_gdImageCreateFromGifPtr )
      ->addParam( "size" )->addParam( "data" );
   self->addClassMethod( sym_GdImage, "CharUp", &_falbind_GdImage_CharUp ).asSymbol()
      ->addParam( "im" )->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "c" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "SetClip", &_falbind_GdImage_SetClip ).asSymbol()
      ->addParam( "im" )->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" );
   self->addExtFunc( "gdImageCreateFrom(WBMP)Ptr", _falbind_gdImageCreateFromWBMPPtr )
      ->addParam( "size" )->addParam( "data" );
   self->addClassMethod( sym_GdImage, "StringUp16", &_falbind_GdImage_StringUp16 ).asSymbol()
      ->addParam( "im" )->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "s" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "SetPixel", &_falbind_GdImage_SetPixel ).asSymbol()
      ->addParam( "im" )->addParam( "x" )->addParam( "y" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "String", &_falbind_GdImage_String ).asSymbol()
      ->addParam( "im" )->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "s" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "FilledArc", &_falbind_GdImage_FilledArc ).asSymbol()
      ->addParam( "im" )->addParam( "cx" )->addParam( "cy" )->addParam( "w" )->addParam( "h" )->addParam( "s" )->addParam( "e" )->addParam( "color" )->addParam( "style" );
   self->addClassMethod( sym_GdImage, "WBMPPtr", &_falbind_GdImage_WBMPPtr ).asSymbol()
      ->addParam( "im" )->addParam( "size" )->addParam( "fg" );
   self->addClassMethod( sym_GdImage, "ColorExact", &_falbind_GdImage_ColorExact ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "Gd2Ptr", &_falbind_GdImage_Gd2Ptr ).asSymbol()
      ->addParam( "im" )->addParam( "cs" )->addParam( "fmt" )->addParam( "size" );
   self->addClassMethod( sym_GdImage, "SetAntiAliasedDontBlend", &_falbind_GdImage_SetAntiAliasedDontBlend ).asSymbol()
      ->addParam( "im" )->addParam( "c" )->addParam( "dont_blend" );
   self->addClassMethod( sym_GdImage, "StringUp", &_falbind_GdImage_StringUp ).asSymbol()
      ->addParam( "im" )->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "s" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "CreatePaletteFromTrueColor", &_falbind_GdImage_CreatePaletteFromTrueColor ).asSymbol()
      ->addParam( "im" )->addParam( "ditherFlag" )->addParam( "colorsWanted" );
   self->addClassMethod( sym_GdImage, "GetPixel", &_falbind_GdImage_GetPixel ).asSymbol()
      ->addParam( "im" )->addParam( "x" )->addParam( "y" );
   self->addClassMethod( sym_GdImage, "ColorExactAlpha", &_falbind_GdImage_ColorExactAlpha ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" )->addParam( "a" );
   self->addClassMethod( sym_GdImage, "GifAnimAddPtr", &_falbind_GdImage_GifAnimAddPtr ).asSymbol()
      ->addParam( "im" )->addParam( "size" )->addParam( "LocalCM" )->addParam( "LeftOfs" )->addParam( "TopOfs" )->addParam( "Delay" )->addParam( "Disposal" )->addParam( "previm" );
   self->addClassMethod( sym_GdImage, "Rectangle", &_falbind_GdImage_Rectangle ).asSymbol()
      ->addParam( "im" )->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "GdPtr", &_falbind_GdImage_GdPtr ).asSymbol()
      ->addParam( "im" )->addParam( "size" );
   self->addClassMethod( sym_GdImage, "Arc", &_falbind_GdImage_Arc ).asSymbol()
      ->addParam( "im" )->addParam( "cx" )->addParam( "cy" )->addParam( "w" )->addParam( "h" )->addParam( "s" )->addParam( "e" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "ColorResolveAlpha", &_falbind_GdImage_ColorResolveAlpha ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" )->addParam( "a" );
   self->addClassMethod( sym_GdImage, "ColorAllocate", &_falbind_GdImage_ColorAllocate ).asSymbol()
      ->addParam( "im" )->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "DashedLine", &_falbind_GdImage_DashedLine ).asSymbol()
      ->addParam( "im" )->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "FilledRectangle", &_falbind_GdImage_FilledRectangle ).asSymbol()
      ->addParam( "im" )->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "CopyRotated", &_falbind_GdImage_CopyRotated ).asSymbol()
      ->addParam( "dst" )->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "srcWidth" )->addParam( "srcHeight" )->addParam( "angle" );
   return self;
}

/* Done with falbind */
