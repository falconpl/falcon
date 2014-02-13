extern "C" {
   #include <gd.h>
   #include <gdfontt.h>
   #include <gdfonts.h>
   #include <gdfontmb.h>
   #include <gdfontl.h>
   #include <gdfontg.h>
}

/********************************************************
  Falcon Stream to gd bridge
*********************************************************/

#include <falcon/memory.h>
#include <falcon/stream.h>
#include <falcon/error.h>

#ifndef FALCON_ERROR_GD_BASE
   #define FALCON_ERROR_GD_BASE 2330
#endif

typedef struct tag_Stream_gdIOCtx
{
   gdIOCtx ctx;
   Falcon::Stream* stream; 
   bool okToDelete;
} StreamCtx;


static int StreamIOCtx_getC(struct gdIOCtx *ctx)
{
   StreamCtx* sctx = (StreamCtx*) ctx;
   Falcon::uint32 val;
   if ( ! sctx->stream->get( val ) )
      return -1;
      
   return (int) val;
}


static int StreamIOCtx_getBuf( struct gdIOCtx *ctx, void *data, int wanted )
{
   StreamCtx* sctx = (StreamCtx*) ctx;
   if ( sctx->stream->eof() )
      return 0;

   return (int) sctx->stream->read( (Falcon::byte*)data, wanted );
}

static void StreamIOCtx_putC( struct gdIOCtx *ctx, int c)
{
   StreamCtx* sctx = (StreamCtx*) ctx;
   sctx->stream->put( c );
}


static int StreamIOCtx_putBuf( struct gdIOCtx *ctx, const void *data, int wanted)
{
   StreamCtx* sctx = (StreamCtx*) ctx;
   return sctx->stream->write( (Falcon::byte*) data, wanted );
}

static int StreamIOCtx_seek(struct gdIOCtx *ctx, const int pos)
{
   StreamCtx* sctx = (StreamCtx*) ctx;
   return sctx->stream->seekBegin( pos );
}

static long StreamIOCtx_tell(struct gdIOCtx *ctx )
{
   StreamCtx* sctx = (StreamCtx*) ctx;
   return (long) sctx->stream->tell();
}

static void StreamIOCtx_free(struct gdIOCtx *ctx)
{
   StreamCtx* sctx = (StreamCtx*) ctx;
   if ( sctx->okToDelete )
       delete sctx->stream;
   Falcon::free( sctx );
}

static gdIOCtx* CreateStreamIOCtx( Falcon::Stream* stream, bool okToDelete )
{
   StreamCtx* sctx = (StreamCtx*) Falcon::malloc( sizeof( StreamCtx ) );
   sctx->ctx.getC = StreamIOCtx_getC;
   sctx->ctx.getBuf = StreamIOCtx_getBuf;
   sctx->ctx.putC = StreamIOCtx_putC;
   sctx->ctx.putBuf = StreamIOCtx_putBuf;
   sctx->ctx.seek = StreamIOCtx_seek;
   sctx->ctx.tell = StreamIOCtx_tell;
   sctx->ctx.gd_free = StreamIOCtx_free;

   sctx->stream = stream;
   sctx->okToDelete = okToDelete;
   return (gdIOCtx*) sctx;
}

#include <falcon/engine.h>
#include <falcon/autocstring.h>
#include <falcon/autowstring.h>

using namespace Falcon;


class _falbind_GdPoint: public CoreObject
{
public:
   _falbind_GdPoint(const CoreClass* cgen, void* ud, bool):
      CoreObject( cgen )
   {
      setUserData( ud );
   }
   
   ~_falbind_GdPoint() {
      
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

static CoreObject* _falbind_GdPoint_factory( const CoreClass* cgen, void* ud, bool bDeser )
{
   if ( ud == 0 )
      throw new CodeError( ErrorParam( e_non_callable, __LINE__ )
         .extra( "Opaque class instantiated" ) );
         
   return new _falbind_GdPoint(cgen, ud, bDeser);
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


class _falbind_gdFTStringExtra: public CoreObject
{
public:
   _falbind_gdFTStringExtra(const CoreClass* cgen, void* ud, bool):
      CoreObject( cgen )
   {
      setUserData( ud );
   }
   
   ~_falbind_gdFTStringExtra() {
      
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

static CoreObject* _falbind_gdFTStringExtra_factory( const CoreClass* cgen, void* ud, bool bDeser )
{
   if ( ud == 0 )
      throw new CodeError( ErrorParam( e_non_callable, __LINE__ )
         .extra( "Opaque class instantiated" ) );
         
   return new _falbind_gdFTStringExtra(cgen, ud, bDeser);
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
   virtual _falbind_GdImage* clone() const {
      return 0;
   }

   gdImage* get() const { return static_cast<gdImage*>( this->getUserData() ); }
};

static CoreObject* _falbind_GdImage_factory( const CoreClass* cgen, void* ud, bool bDeser )
{
   return new _falbind_GdImage(cgen, ud, bDeser);
}

/**************************************
   Custom error class GdError
***************************************/

class GdError: public ::Falcon::Error
{
public:
   GdError():
      Error( "GdError" )
   {}

   GdError( const ErrorParam &params  ):
      Error( "GdError", params )
      {}
};

static void  GdError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new GdError );

   ::Falcon::core::Error_init( vm );
}

/**************************************
   End of GdError
***************************************/


static void _falbind_GdImage_PaletteCopy( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage" ) );
   }

   gdImageStruct* dst = self->get();
gdImage* src = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImagePaletteCopy( dst, src );
}


static void _falbind_GdImage_Red( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int color = (int) vm->param(0)->forceInteger();
   int __funcreturn__ = gdImageRed( im, color );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_Rectangle( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x1 = (int) vm->param(0)->forceInteger();
   int y1 = (int) vm->param(1)->forceInteger();
   int x2 = (int) vm->param(2)->forceInteger();
   int y2 = (int) vm->param(3)->forceInteger();
   int color = (int) vm->param(4)->forceInteger();
gdImageRectangle( im, x1, y1, x2, y2, color );
}


static void _falbind_GdImage_Line( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x1 = (int) vm->param(0)->forceInteger();
   int y1 = (int) vm->param(1)->forceInteger();
   int x2 = (int) vm->param(2)->forceInteger();
   int y2 = (int) vm->param(3)->forceInteger();
   int color = (int) vm->param(4)->forceInteger();
gdImageLine( im, x1, y1, x2, y2, color );
}


static void _falbind_GdImage_AlphaBlending( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int alphaBlendingArg = (int) vm->param(0)->forceInteger();
gdImageAlphaBlending( im, alphaBlendingArg );
}


static void _falbind_GdImage_Blue( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int color = (int) vm->param(0)->forceInteger();
   int __funcreturn__ = gdImageBlue( im, color );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_Char( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdFont,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
gdFont* f = (gdFont*) vm->param(0)->asObject()->getUserData();
   int x = (int) vm->param(1)->forceInteger();
   int y = (int) vm->param(2)->forceInteger();
   int c = (int) vm->param(3)->forceInteger();
   int color = (int) vm->param(4)->forceInteger();
gdImageChar( im, f, x, y, c, color );
}


static void _falbind_GdImage_DashedLine( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x1 = (int) vm->param(0)->forceInteger();
   int y1 = (int) vm->param(1)->forceInteger();
   int x2 = (int) vm->param(2)->forceInteger();
   int y2 = (int) vm->param(3)->forceInteger();
   int color = (int) vm->param(4)->forceInteger();
gdImageDashedLine( im, x1, y1, x2, y2, color );
}


static void _falbind_gdImageTrueColor( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage" ) );
   }

gdImage* im = (gdImage*) vm->param(0)->asObject()->getUserData();
   int __funcreturn__ = gdImageTrueColor( im );

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+2, __LINE__ )
         .desc( "Error in creating the image" ) );
   }
   
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_gdFontGetLarge( VMachine *vm )
{
   gdFont* __funcreturn__ = gdFontGetLarge(  );
   CoreObject* co___funcreturn__ = vm->findWKI( "GdFont" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_Jpeg( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream,N" ) );
   }

   gdImageStruct* im = self->get();
   gdIOCtx* out = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   int quality = (int) vm->param(1)->forceInteger();
gdImageJpegCtx( im, out, quality );
   out->gd_free(out);

   if( !dyncast<Stream*>(vm->param(0)->asObject()->getFalconData())->good() ) {
      throw new IoError( ErrorParam( FALCON_ERROR_GD_BASE+3, __LINE__ )
         .desc( "I/O error while writing the image" ) );
   }
   
}


static void _falbind_GdImage_SetBrush( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage" ) );
   }

   gdImageStruct* im = self->get();
gdImage* brush = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImageSetBrush( im, brush );
}


static void _falbind_GdImage_Png( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdImageStruct* im = self->get();
   gdIOCtx* out = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
gdImagePngCtx( im, out );
   out->gd_free(out);

   if( !dyncast<Stream*>(vm->param(0)->asObject()->getFalconData())->good() ) {
      throw new IoError( ErrorParam( FALCON_ERROR_GD_BASE+3, __LINE__ )
         .desc( "I/O error while writing the image" ) );
   }
   
}


static void _falbind_GdImage_ColorClosestHWB( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int __funcreturn__ = gdImageColorClosestHWB( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_GetPixel( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x = (int) vm->param(0)->forceInteger();
   int y = (int) vm->param(1)->forceInteger();
   int __funcreturn__ = gdImageGetPixel( im, x, y );
   vm->retval( (int64)__funcreturn__);
}


/* Custom binding for gdImageStringFT/FTEx */

static void _falbind_GDImage_stringFT( Falcon::VMachine* vm )
{
   // parameter retrival
   Falcon::Item *i_fg = vm->param(0);
   Falcon::Item *i_fontname = vm->param(1);
   Falcon::Item *i_ptsize = vm->param(2);
   Falcon::Item *i_angle = vm->param(3);
   Falcon::Item *i_x = vm->param(4);
   Falcon::Item *i_y = vm->param(5);
   Falcon::Item *i_string = vm->param(6);
   Falcon::Item *i_extra = vm->param(7);  // optional parameter for FtEX

   if( i_fg == 0 || ! i_fg->isOrdinal() ||
       i_fontname == 0 || ! i_fontname->isString() ||
       i_ptsize == 0 || !( i_ptsize->isNil() || i_ptsize->isOrdinal() ) ||
       i_angle == 0 || !( i_angle->isNil() || i_angle->isOrdinal() ) ||
       i_x == 0 || !i_x->isOrdinal() ||
       i_y == 0 || !i_y->isOrdinal() ||
       i_string == 0 || ! i_string->isString() ||
       (i_extra != 0 && ! (i_extra->isBoolean() || i_extra->isDict() ) )
       )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,S,N,N,N,N,S,[B|D]" ) );
   }

   // a bit of data to be readied.

   gdImagePtr img;
   bool return_brect;
   bool extra;
   int brect[8];

   int fg = i_fg->forceInteger();
   Falcon::AutoCString fname( *i_fontname->asString() );
   const char* fontname = fname.c_str();
   int ptsize = i_ptsize->forceInteger();
   int angle = i_angle->forceInteger();
   int x = i_x->forceInteger();
   int y = i_y->forceInteger();
   Falcon::AutoCString str( *i_string->asString() );
   const char* string = str.c_str();

   char* res;  // if zero, we failed.

   // first, determine if we're called statically -- in this case, we use
   // NULL parameter for img, to get a dry execution for bounds calculation.
   if ( vm->self().isObject() )
   {
      _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );
      img = self->get();
      return_brect = i_extra != 0 && i_extra->isBoolean() && i_extra->asBoolean();
   }
   else {
      // called as static -- dry execution.
      img = 0;
      return_brect = true;
   }

   extra = i_extra != 0 && i_extra->isDict();

   // we're ready for the call
   if( extra )
   {
      Falcon::CoreDict* d = i_extra->asDict();
      Falcon::AutoCString* cs_xshow = 0;
      Falcon::AutoCString* cs_fontpath = 0;
      
      // we must extract extra parameters.
      gdFTStringExtra xp;
      Falcon::Item* i_flags = d->find( "flags" );
      xp.flags = (int)(i_flags != 0 ? i_flags->forceInteger() : 0);
      Falcon::Item* i_spacing = d->find( "linespacing" );
      xp.linespacing = i_spacing != 0 ? i_spacing->forceNumeric() : 0.0;
      Falcon::Item* i_charmap = d->find( "charmap" );
      xp.charmap = (int) (i_charmap != 0 ? i_charmap->forceInteger() : 0);
      Falcon::Item* i_hdpi = d->find( "hdpi" );
      xp.hdpi = (int) (i_hdpi != 0 ? i_hdpi->forceInteger() : 0);
      Falcon::Item* i_vdpi = d->find( "vdpi" );
      xp.vdpi = (int) (i_hdpi != 0 ? i_vdpi->forceInteger() : 0);

      Item* i_xshow = d->find( "xshow" );
      if ( i_xshow->isString() )
      {
         cs_xshow = new AutoCString( *i_xshow->asString() );
         xp.xshow = const_cast<char*>(cs_xshow->c_str());
      }
      else
         xp.xshow = 0;
         
      Item* i_fontpath = d->find( "fontpath" );
      if ( i_fontpath->isString() )
      {
         cs_fontpath = new AutoCString( *i_fontpath->asString() );
         xp.fontpath = const_cast<char*>(cs_fontpath->c_str());
      }
      else
         xp.xshow = 0;

      // finally, we got to determine if we need to return brecht
      if ( img != 0 && d->find( "brect" ) != 0 )
         return_brect = true;

      // we can make the call
      res = gdImageStringFTEx( img, brect, fg, const_cast<char*>(fontname), ptsize, angle, x, y, const_cast<char*>(string), &xp );

      delete cs_xshow;
      delete cs_fontpath;
   }
   else {
      // no, it's a standard call
      res = gdImageStringFT( img, brect, fg, const_cast<char*>(fontname), ptsize, angle, x, y, const_cast<char*>(string) );
   }

   // error?
   if ( res != 0 )
   {
      throw new GdError(
         Falcon::ErrorParam( FALCON_ERROR_GD_BASE, __LINE__ )
         .desc( "Error in StringFT" )
         .extra( res ) );
   }
   if ( return_brect )
   {
      Falcon::CoreArray* ca = new Falcon::CoreArray( 8 );
      for ( Falcon::uint32 i = 0; i < 8; i ++ )
         ca->append( (Falcon::int64) brect[i] );
      vm->retval( ca );
   }
   else
      vm->retnil();
}

static void _falbind_GdImage_CreateFromGd2( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdIOCtx* in = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   gdImage* __funcreturn__ = gdImageCreateFromGd2Ctx( in );
   in->gd_free(in);

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+1, __LINE__ )
         .desc( "Invalid image format" ) );
   }
   
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_StringUp( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isString()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdFont,N,N,S,N" ) );
   }

   gdImageStruct* im = self->get();
gdFont* f = (gdFont*) vm->param(0)->asObject()->getUserData();
   int x = (int) vm->param(1)->forceInteger();
   int y = (int) vm->param(2)->forceInteger();
   AutoCString autoc_s( *vm->param(3)->asString() );
   unsigned char* s = (unsigned char*) autoc_s.c_str();
   int color = (int) vm->param(4)->forceInteger();
gdImageStringUp( im, f, x, y, s, color );
}


static void _falbind_gdFontGetGiant( VMachine *vm )
{
   gdFont* __funcreturn__ = gdFontGetGiant(  );
   CoreObject* co___funcreturn__ = vm->findWKI( "GdFont" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_CreateFromGd( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdIOCtx* in = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   gdImage* __funcreturn__ = gdImageCreateFromGdCtx( in );
   in->gd_free(in);

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+1, __LINE__ )
         .desc( "Invalid image format" ) );
   }
   
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_ColorResolve( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int __funcreturn__ = gdImageColorResolve( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_SaveAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int saveAlphaArg = (int) vm->param(0)->forceInteger();
gdImageSaveAlpha( im, saveAlphaArg );
}


static void _falbind_GdImage_Copy( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage,N,N,N,N,N,N" ) );
   }

   gdImageStruct* dst = self->get();
gdImage* src = (gdImage*) vm->param(0)->asObject()->getUserData();
   int dstX = (int) vm->param(1)->forceInteger();
   int dstY = (int) vm->param(2)->forceInteger();
   int srcX = (int) vm->param(3)->forceInteger();
   int srcY = (int) vm->param(4)->forceInteger();
   int w = (int) vm->param(5)->forceInteger();
   int h = (int) vm->param(6)->forceInteger();
gdImageCopy( dst, src, dstX, dstY, srcX, srcY, w, h );
}


static void _falbind_GdImage_SetAntiAliasedDontBlend( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

   gdImageStruct* im = self->get();
   int c = (int) vm->param(0)->forceInteger();
   int dont_blend = (int) vm->param(1)->forceInteger();
gdImageSetAntiAliasedDontBlend( im, c, dont_blend );
}


static void _falbind_GdImage_init( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

   int sx = (int) vm->param(0)->forceInteger();
   int sy = (int) vm->param(1)->forceInteger();
   gdImage* __funcreturn__ = gdImageCreate( sx, sy );

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+2, __LINE__ )
         .desc( "Error in creating the image" ) );
   }
   
   vm->self().asObject()->setUserData( __funcreturn__ );
}


static void _falbind_GdImage_CopyMergeGray( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
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

   gdImageStruct* dst = self->get();
gdImage* src = (gdImage*) vm->param(0)->asObject()->getUserData();
   int dstX = (int) vm->param(1)->forceInteger();
   int dstY = (int) vm->param(2)->forceInteger();
   int srcX = (int) vm->param(3)->forceInteger();
   int srcY = (int) vm->param(4)->forceInteger();
   int w = (int) vm->param(5)->forceInteger();
   int h = (int) vm->param(6)->forceInteger();
   int pct = (int) vm->param(7)->forceInteger();
gdImageCopyMergeGray( dst, src, dstX, dstY, srcX, srcY, w, h, pct );
}


static void _falbind_GdImage_Fill( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x = (int) vm->param(0)->forceInteger();
   int y = (int) vm->param(1)->forceInteger();
   int color = (int) vm->param(2)->forceInteger();
gdImageFill( im, x, y, color );
}


static void _falbind_GdImage_WBMP( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,Stream" ) );
   }

   gdImageStruct* image = self->get();
   int fg = (int) vm->param(0)->forceInteger();
   gdIOCtx* out = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(1)->asObject()->getFalconData()),
         false );
gdImageWBMPCtx( image, fg, out );
   out->gd_free(out);

   if( !dyncast<Stream*>(vm->param(1)->asObject()->getFalconData())->good() ) {
      throw new IoError( ErrorParam( FALCON_ERROR_GD_BASE+3, __LINE__ )
         .desc( "I/O error while writing the image" ) );
   }
   
}


static void _falbind_GdImage_GifAnimBegin( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream,N,N" ) );
   }

   gdImageStruct* im = self->get();
   gdIOCtx* out = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   int GlobalCM = (int) vm->param(1)->forceInteger();
   int Loops = (int) vm->param(2)->forceInteger();
gdImageGifAnimBeginCtx( im, out, GlobalCM, Loops );
   out->gd_free(out);

   if( !dyncast<Stream*>(vm->param(0)->asObject()->getFalconData())->good() ) {
      throw new IoError( ErrorParam( FALCON_ERROR_GD_BASE+3, __LINE__ )
         .desc( "I/O error while writing the image" ) );
   }
   
}


static void _falbind_GdImage_ColorClosestAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int a = (int) vm->param(3)->forceInteger();
   int __funcreturn__ = gdImageColorClosestAlpha( im, r, g, b, a );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_gdFontGetTiny( VMachine *vm )
{
   gdFont* __funcreturn__ = gdFontGetTiny(  );
   CoreObject* co___funcreturn__ = vm->findWKI( "GdFont" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_SetClip( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x1 = (int) vm->param(0)->forceInteger();
   int y1 = (int) vm->param(1)->forceInteger();
   int x2 = (int) vm->param(2)->forceInteger();
   int y2 = (int) vm->param(3)->forceInteger();
gdImageSetClip( im, x1, y1, x2, y2 );
}


static void _falbind_GdImage_ColorAllocate( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int __funcreturn__ = gdImageColorAllocate( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_ColorAllocateAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int a = (int) vm->param(3)->forceInteger();
   int __funcreturn__ = gdImageColorAllocateAlpha( im, r, g, b, a );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_CopyResized( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
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

   gdImageStruct* dst = self->get();
gdImage* src = (gdImage*) vm->param(0)->asObject()->getUserData();
   int dstX = (int) vm->param(1)->forceInteger();
   int dstY = (int) vm->param(2)->forceInteger();
   int srcX = (int) vm->param(3)->forceInteger();
   int srcY = (int) vm->param(4)->forceInteger();
   int dstW = (int) vm->param(5)->forceInteger();
   int dstH = (int) vm->param(6)->forceInteger();
   int srcW = (int) vm->param(7)->forceInteger();
   int srcH = (int) vm->param(8)->forceInteger();
gdImageCopyResized( dst, src, dstX, dstY, srcX, srcY, dstW, dstH, srcW, srcH );
}


static void _falbind_GdImage_CreateFromGd2Part( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream,N,N,N,N" ) );
   }

   gdIOCtx* in = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   int srcx = (int) vm->param(1)->forceInteger();
   int srcy = (int) vm->param(2)->forceInteger();
   int w = (int) vm->param(3)->forceInteger();
   int h = (int) vm->param(4)->forceInteger();
   gdImage* __funcreturn__ = gdImageCreateFromGd2PartCtx( in, srcx, srcy, w, h );
   in->gd_free(in);

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+1, __LINE__ )
         .desc( "Invalid image format" ) );
   }
   
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_SetStyle( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   (vm->param(0)->isOrdinal() && vm->isParamByRef(0))))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "$N,N" ) );
   }

   gdImageStruct* im = self->get();
  int style = (int) vm->param(0)->forceInteger();
   int noOfPixels = (int) vm->param(1)->forceInteger();
gdImageSetStyle( im,    &style, noOfPixels );
   *vm->param(0) = (int64) style;
}


static void _falbind_GdImage_FilledArc( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
        ||( vm->param(7) == 0 || !(   vm->param(7)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int cx = (int) vm->param(0)->forceInteger();
   int cy = (int) vm->param(1)->forceInteger();
   int w = (int) vm->param(2)->forceInteger();
   int h = (int) vm->param(3)->forceInteger();
   int s = (int) vm->param(4)->forceInteger();
   int e = (int) vm->param(5)->forceInteger();
   int color = (int) vm->param(6)->forceInteger();
   int style = (int) vm->param(7)->forceInteger();
gdImageFilledArc( im, cx, cy, w, h, s, e, color, style );
}


static void _falbind_gdFontGetSmall( VMachine *vm )
{
   gdFont* __funcreturn__ = gdFontGetSmall(  );
   CoreObject* co___funcreturn__ = vm->findWKI( "GdFont" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_SetTile( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage" ) );
   }

   gdImageStruct* im = self->get();
gdImage* tile = (gdImage*) vm->param(0)->asObject()->getUserData();
gdImageSetTile( im, tile );
}


static void _falbind_GdImage_PngEx( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream,N" ) );
   }

   gdImageStruct* im = self->get();
   gdIOCtx* out = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   int level = (int) vm->param(1)->forceInteger();
gdImagePngCtxEx( im, out, level );
   out->gd_free(out);

   if( !dyncast<Stream*>(vm->param(0)->asObject()->getFalconData())->good() ) {
      throw new IoError( ErrorParam( FALCON_ERROR_GD_BASE+3, __LINE__ )
         .desc( "I/O error while writing the image" ) );
   }
   
}


static void _falbind_GdImage_GetClip( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   (vm->param(0)->isOrdinal() && vm->isParamByRef(0))))
        ||( vm->param(1) == 0 || !(   (vm->param(1)->isOrdinal() && vm->isParamByRef(1))))
        ||( vm->param(2) == 0 || !(   (vm->param(2)->isOrdinal() && vm->isParamByRef(2))))
        ||( vm->param(3) == 0 || !(   (vm->param(3)->isOrdinal() && vm->isParamByRef(3))))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "$N,$N,$N,$N" ) );
   }

   gdImageStruct* im = self->get();
  int x1P = (int) vm->param(0)->forceInteger();
  int y1P = (int) vm->param(1)->forceInteger();
  int x2P = (int) vm->param(2)->forceInteger();
  int y2P = (int) vm->param(3)->forceInteger();
gdImageGetClip( im,    &x1P,    &y1P,    &x2P,    &y2P );
   *vm->param(0) = (int64) x1P;
   *vm->param(1) = (int64) y1P;
   *vm->param(2) = (int64) x2P;
   *vm->param(3) = (int64) y2P;
}


static void _falbind_GdImage_ColorTransparent( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int color = (int) vm->param(0)->forceInteger();
gdImageColorTransparent( im, color );
}


static void _falbind_GdImage_AABlend( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   gdImageStruct* im = self->get();
gdImageAABlend( im );
}


static void _falbind_GdImage_SetAntiAliased( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int c = (int) vm->param(0)->forceInteger();
gdImageSetAntiAliased( im, c );
}


static void _falbind_GdImage_Compare( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdImage" ) );
   }

   gdImageStruct* im1 = self->get();
gdImage* im2 = (gdImage*) vm->param(0)->asObject()->getUserData();
   int __funcreturn__ = gdImageCompare( im1, im2 );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_CharUp( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdFont,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
gdFont* f = (gdFont*) vm->param(0)->asObject()->getUserData();
   int x = (int) vm->param(1)->forceInteger();
   int y = (int) vm->param(2)->forceInteger();
   int c = (int) vm->param(3)->forceInteger();
   int color = (int) vm->param(4)->forceInteger();
gdImageCharUp( im, f, x, y, c, color );
}


static void _falbind_GdImage_CopyRotated( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
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

   gdImageStruct* dst = self->get();
gdImage* src = (gdImage*) vm->param(0)->asObject()->getUserData();
double dstX = (double) vm->param(1)->forceNumeric();
double dstY = (double) vm->param(2)->forceNumeric();
   int srcX = (int) vm->param(3)->forceInteger();
   int srcY = (int) vm->param(4)->forceInteger();
   int srcWidth = (int) vm->param(5)->forceInteger();
   int srcHeight = (int) vm->param(6)->forceInteger();
   int angle = (int) vm->param(7)->forceInteger();
gdImageCopyRotated( dst, src, dstX, dstY, srcX, srcY, srcWidth, srcHeight, angle );
}


static void _falbind_gdFontGetMediumBold( VMachine *vm )
{
   gdFont* __funcreturn__ = gdFontGetMediumBold(  );
   CoreObject* co___funcreturn__ = vm->findWKI( "GdFont" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_TrueColorToPalette( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

   gdImageStruct* im = self->get();
   int ditherFlag = (int) vm->param(0)->forceInteger();
   int colorsWanted = (int) vm->param(1)->forceInteger();
gdImageTrueColorToPalette( im, ditherFlag, colorsWanted );
}


static void _falbind_GdImage_CreateFromJpeg( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdIOCtx* infile = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   gdImage* __funcreturn__ = gdImageCreateFromJpegCtx( infile );
   infile->gd_free(infile);

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+1, __LINE__ )
         .desc( "Invalid image format" ) );
   }
   
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_CreateFromPng( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdIOCtx* in = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   gdImage* __funcreturn__ = gdImageCreateFromPngCtx( in );
   in->gd_free(in);

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+1, __LINE__ )
         .desc( "Invalid image format" ) );
   }
   
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_FilledRectangle( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x1 = (int) vm->param(0)->forceInteger();
   int y1 = (int) vm->param(1)->forceInteger();
   int x2 = (int) vm->param(2)->forceInteger();
   int y2 = (int) vm->param(3)->forceInteger();
   int color = (int) vm->param(4)->forceInteger();
gdImageFilledRectangle( im, x1, y1, x2, y2, color );
}


static void _falbind_GdImage_SetPixel( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x = (int) vm->param(0)->forceInteger();
   int y = (int) vm->param(1)->forceInteger();
   int color = (int) vm->param(2)->forceInteger();
gdImageSetPixel( im, x, y, color );
}


static void _falbind_GdImage_BoundsSafe( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x = (int) vm->param(0)->forceInteger();
   int y = (int) vm->param(1)->forceInteger();
   int __funcreturn__ = gdImageBoundsSafe( im, x, y );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_Polygon( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdPoint")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdPoint,N,N" ) );
   }

   gdImageStruct* im = self->get();
gdPoint* p = (gdPoint*) vm->param(0)->asObject()->getUserData();
   int n = (int) vm->param(1)->forceInteger();
   int c = (int) vm->param(2)->forceInteger();
gdImagePolygon( im, p, n, c );
}


static void _falbind_GdImage_GifAnimEnd( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdIOCtx* out = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
gdImageGifAnimEndCtx( out );
   out->gd_free(out);

   if( !dyncast<Stream*>(vm->param(0)->asObject()->getFalconData())->good() ) {
      throw new IoError( ErrorParam( FALCON_ERROR_GD_BASE+3, __LINE__ )
         .desc( "I/O error while writing the image" ) );
   }
   
}


static void _falbind_GdImage_Arc( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int cx = (int) vm->param(0)->forceInteger();
   int cy = (int) vm->param(1)->forceInteger();
   int w = (int) vm->param(2)->forceInteger();
   int h = (int) vm->param(3)->forceInteger();
   int s = (int) vm->param(4)->forceInteger();
   int e = (int) vm->param(5)->forceInteger();
   int color = (int) vm->param(6)->forceInteger();
gdImageArc( im, cx, cy, w, h, s, e, color );
}


static void _falbind_GdImage_CopyMerge( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
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

   gdImageStruct* dst = self->get();
gdImage* src = (gdImage*) vm->param(0)->asObject()->getUserData();
   int dstX = (int) vm->param(1)->forceInteger();
   int dstY = (int) vm->param(2)->forceInteger();
   int srcX = (int) vm->param(3)->forceInteger();
   int srcY = (int) vm->param(4)->forceInteger();
   int w = (int) vm->param(5)->forceInteger();
   int h = (int) vm->param(6)->forceInteger();
   int pct = (int) vm->param(7)->forceInteger();
gdImageCopyMerge( dst, src, dstX, dstY, srcX, srcY, w, h, pct );
}


static void _falbind_GdImage_CreateFromWBMP( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdIOCtx* infile = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   gdImage* __funcreturn__ = gdImageCreateFromWBMPCtx( infile );
   infile->gd_free(infile);

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+1, __LINE__ )
         .desc( "Invalid image format" ) );
   }
   
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_GifAnimAdd( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
        ||( vm->param(5) == 0 || !(   vm->param(5)->isOrdinal()))
        ||( vm->param(6) == 0 || !(   vm->param(6)->isObject() && vm->param(6)->asObjectSafe()->derivedFrom("GdImage")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream,N,N,N,N,N,GdImage" ) );
   }

   gdImageStruct* im = self->get();
   gdIOCtx* out = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   int LocalCM = (int) vm->param(1)->forceInteger();
   int LeftOfs = (int) vm->param(2)->forceInteger();
   int TopOfs = (int) vm->param(3)->forceInteger();
   int Delay = (int) vm->param(4)->forceInteger();
   int Disposal = (int) vm->param(5)->forceInteger();
gdImage* previm = (gdImage*) vm->param(6)->asObject()->getUserData();
gdImageGifAnimAddCtx( im, out, LocalCM, LeftOfs, TopOfs, Delay, Disposal, previm );
   out->gd_free(out);

   if( !dyncast<Stream*>(vm->param(0)->asObject()->getFalconData())->good() ) {
      throw new IoError( ErrorParam( FALCON_ERROR_GD_BASE+3, __LINE__ )
         .desc( "I/O error while writing the image" ) );
   }
   
}


static void _falbind_GdImage_SetThickness( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int thickness = (int) vm->param(0)->forceInteger();
gdImageSetThickness( im, thickness );
}


static void _falbind_GdImage_FilledPolygon( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdPoint")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdPoint,N,N" ) );
   }

   gdImageStruct* im = self->get();
gdPoint* p = (gdPoint*) vm->param(0)->asObject()->getUserData();
   int n = (int) vm->param(1)->forceInteger();
   int c = (int) vm->param(2)->forceInteger();
gdImageFilledPolygon( im, p, n, c );
}


static void _falbind_GdImage_ColorDeallocate( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int color = (int) vm->param(0)->forceInteger();
gdImageColorDeallocate( im, color );
}


static void _falbind_GdImage_Interlace( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int interlaceArg = (int) vm->param(0)->forceInteger();
gdImageInterlace( im, interlaceArg );
}


static void _falbind_GdImage_FilledEllipse( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int cx = (int) vm->param(0)->forceInteger();
   int cy = (int) vm->param(1)->forceInteger();
   int w = (int) vm->param(2)->forceInteger();
   int h = (int) vm->param(3)->forceInteger();
   int color = (int) vm->param(4)->forceInteger();
gdImageFilledEllipse( im, cx, cy, w, h, color );
}


static void _falbind_GdImage_CreateTrueColor( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

   int sx = (int) vm->param(0)->forceInteger();
   int sy = (int) vm->param(1)->forceInteger();
   gdImage* __funcreturn__ = gdImageCreateTrueColor( sx, sy );
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_FillToBorder( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x = (int) vm->param(0)->forceInteger();
   int y = (int) vm->param(1)->forceInteger();
   int border = (int) vm->param(2)->forceInteger();
   int color = (int) vm->param(3)->forceInteger();
gdImageFillToBorder( im, x, y, border, color );
}


static void _falbind_GdImage_OpenPolygon( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdPoint")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdPoint,N,N" ) );
   }

   gdImageStruct* im = self->get();
gdPoint* p = (gdPoint*) vm->param(0)->asObject()->getUserData();
   int n = (int) vm->param(1)->forceInteger();
   int c = (int) vm->param(2)->forceInteger();
gdImageOpenPolygon( im, p, n, c );
}


static void _falbind_GdImage_SX( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   gdImageStruct* im = self->get();
   int __funcreturn__ = gdImageSX( im );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_SY( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   gdImageStruct* im = self->get();
   int __funcreturn__ = gdImageSY( im );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_ColorClosest( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int __funcreturn__ = gdImageColorClosest( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_CreatePaletteFromTrueColor( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

   gdImageStruct* im = self->get();
   int ditherFlag = (int) vm->param(0)->forceInteger();
   int colorsWanted = (int) vm->param(1)->forceInteger();
   gdImage* __funcreturn__ = gdImageCreatePaletteFromTrueColor( im, ditherFlag, colorsWanted );
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_ColorResolveAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int a = (int) vm->param(3)->forceInteger();
   int __funcreturn__ = gdImageColorResolveAlpha( im, r, g, b, a );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_String( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isString()))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdFont,N,N,S,N" ) );
   }

   gdImageStruct* im = self->get();
gdFont* f = (gdFont*) vm->param(0)->asObject()->getUserData();
   int x = (int) vm->param(1)->forceInteger();
   int y = (int) vm->param(2)->forceInteger();
   AutoCString autoc_s( *vm->param(3)->asString() );
   unsigned char* s = (unsigned char*) autoc_s.c_str();
   int color = (int) vm->param(4)->forceInteger();
gdImageString( im, f, x, y, s, color );
}


static void _falbind_GdImage_Gif( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdImageStruct* im = self->get();
   gdIOCtx* out = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
gdImageGifCtx( im, out );
   out->gd_free(out);

   if( !dyncast<Stream*>(vm->param(0)->asObject()->getFalconData())->good() ) {
      throw new IoError( ErrorParam( FALCON_ERROR_GD_BASE+3, __LINE__ )
         .desc( "I/O error while writing the image" ) );
   }
   
}


static void _falbind_GdImage_ColorExactAlpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   vm->param(3)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int a = (int) vm->param(3)->forceInteger();
   int __funcreturn__ = gdImageColorExactAlpha( im, r, g, b, a );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_CopyResampled( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdImage")))
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

   gdImageStruct* dst = self->get();
gdImage* src = (gdImage*) vm->param(0)->asObject()->getUserData();
   int dstX = (int) vm->param(1)->forceInteger();
   int dstY = (int) vm->param(2)->forceInteger();
   int srcX = (int) vm->param(3)->forceInteger();
   int srcY = (int) vm->param(4)->forceInteger();
   int dstW = (int) vm->param(5)->forceInteger();
   int dstH = (int) vm->param(6)->forceInteger();
   int srcW = (int) vm->param(7)->forceInteger();
   int srcH = (int) vm->param(8)->forceInteger();
gdImageCopyResampled( dst, src, dstX, dstY, srcX, srcY, dstW, dstH, srcW, srcH );
}


static void _falbind_GdImage_Alpha( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int color = (int) vm->param(0)->forceInteger();
   int __funcreturn__ = gdImageAlpha( im, color );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_CreateFromGif( VMachine *vm )
{
   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOfClass("Stream")))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "Stream" ) );
   }

   gdIOCtx* in = (gdIOCtx*) CreateStreamIOCtx(
         dyncast<Stream*>(vm->param(0)->asObject()->getFalconData()),
         false );
   gdImage* __funcreturn__ = gdImageCreateFromGifCtx( in );
   in->gd_free(in);

   if( __funcreturn__ == 0  ) {
      throw new GdError( ErrorParam( FALCON_ERROR_GD_BASE+1, __LINE__ )
         .desc( "Invalid image format" ) );
   }
   
   CoreObject* co___funcreturn__ = vm->findWKI( "GdImage" )->asClass()->createInstance(__funcreturn__);
   vm->retval( co___funcreturn__ );
}


static void _falbind_GdImage_GetTrueColorPixel( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N" ) );
   }

   gdImageStruct* im = self->get();
   int x = (int) vm->param(0)->forceInteger();
   int y = (int) vm->param(1)->forceInteger();
   int __funcreturn__ = gdImageGetTrueColorPixel( im, x, y );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_ColorExact( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N,N,N" ) );
   }

   gdImageStruct* im = self->get();
   int r = (int) vm->param(0)->forceInteger();
   int g = (int) vm->param(1)->forceInteger();
   int b = (int) vm->param(2)->forceInteger();
   int __funcreturn__ = gdImageColorExact( im, r, g, b );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_Green( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "N" ) );
   }

   gdImageStruct* im = self->get();
   int color = (int) vm->param(0)->forceInteger();
   int __funcreturn__ = gdImageGreen( im, color );
   vm->retval( (int64)__funcreturn__);
}


static void _falbind_GdImage_StringUp16( VMachine *vm )
{
   _falbind_GdImage* self = dyncast<_falbind_GdImage*>( vm->self().asObject() );

   if ( ( vm->param(0) == 0 || !(   vm->param(0)->isObject() && vm->param(0)->asObjectSafe()->derivedFrom("GdFont")))
        ||( vm->param(1) == 0 || !(   vm->param(1)->isOrdinal()))
        ||( vm->param(2) == 0 || !(   vm->param(2)->isOrdinal()))
        ||( vm->param(3) == 0 || !(   (vm->param(3)->isOrdinal() && vm->isParamByRef(3))))
        ||( vm->param(4) == 0 || !(   vm->param(4)->isOrdinal()))
   ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "GdFont,N,N,$N,N" ) );
   }

   gdImageStruct* im = self->get();
gdFont* f = (gdFont*) vm->param(0)->asObject()->getUserData();
   int x = (int) vm->param(1)->forceInteger();
   int y = (int) vm->param(2)->forceInteger();
  unsigned short s = (unsigned short) vm->param(3)->forceInteger();
   int color = (int) vm->param(4)->forceInteger();
gdImageStringUp16( im, f, x, y,    &s, color );
   *vm->param(3) = (int64) s;
}


FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self
   Falcon::Module *self = new Falcon::Module();
   
   self->name( "gd2" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( 1, 0, 0 );
   const Symbol*sym_GdPoint = self->addClass("GdPoint");
   sym_GdPoint->setWKS( true );
   sym_GdPoint->getClassDef()->factory( &_falbind_GdPoint_factory );

   const Symbol*sym_GdFont = self->addClass("GdFont");
   sym_GdFont->setWKS( true );
   sym_GdFont->getClassDef()->factory( &_falbind_GdFont_factory );

   const Symbol*sym_gdFTStringExtra = self->addClass("gdFTStringExtra");
   sym_gdFTStringExtra->setWKS( true );
   sym_gdFTStringExtra->getClassDef()->factory( &_falbind_gdFTStringExtra_factory );

   const Symbol*sym_GdImage = self->addClass("GdImage");
   sym_GdImage->setWKS( true );
   sym_GdImage->getClassDef()->factory( &_falbind_GdImage_factory );

   //****************************************
   // Error class GdError
   //
   Falcon::Symbol *GdError_base_error_class = self->addExternalRef( "Error" );
   Falcon::Symbol *GdError_error_class = self->addClass( "GdError", &GdError_init );
   GdError_error_class->setWKS( true );
   GdError_error_class->getClassDef()->addInheritance(  new Falcon::InheritDef( GdError_base_error_class ) );
      self->addConstant( "gdAntiAliased", (int64) gdAntiAliased, true );
   self->addConstant( "gdBrushed", (int64) gdBrushed, true );
   self->addConstant( "gdMaxColors", (int64) gdMaxColors, true );
   self->addConstant( "gdStyled", (int64) gdStyled, true );
   self->addConstant( "gdStyledBrushed", (int64) gdStyledBrushed, true );
   self->addConstant( "gdDashSize", (int64) gdDashSize, true );
   self->addConstant( "gdTiled", (int64) gdTiled, true );
   self->addConstant( "gdArc", (int64) gdArc, true );
   self->addConstant( "gdChord", (int64) gdChord, true );
   self->addConstant( "gdPie", (int64) gdPie, true );
   self->addConstant( "gdNoFill", (int64) gdNoFill, true );
   self->addConstant( "gdEdged", (int64) gdEdged, true );
   self->addConstant( "gdFTEX_LINESPACE", (int64) gdFTEX_LINESPACE, true );
   self->addConstant( "gdFTEX_CHARMAP", (int64) gdFTEX_CHARMAP, true );
   self->addConstant( "gdFTEX_RESOLUTION", (int64) gdFTEX_RESOLUTION, true );
   self->addConstant( "gdFTEX_DISABLE_KERNING", (int64) gdFTEX_DISABLE_KERNING, true );
   self->addConstant( "gdFTEX_XSHOW", (int64) gdFTEX_XSHOW, true );
   self->addConstant( "gdFTEX_RETURNFONTPATHNAME", (int64) gdFTEX_RETURNFONTPATHNAME, true );
   self->addConstant( "gdFTEX_FONTPATHNAME", (int64) gdFTEX_FONTPATHNAME, true );
   self->addConstant( "gdFTEX_FONTCONFIG", (int64) gdFTEX_FONTCONFIG, true );
   self->addClassMethod( sym_GdImage, "PaletteCopy", &_falbind_GdImage_PaletteCopy ).asSymbol()
      ->addParam( "src" );
   self->addClassMethod( sym_GdImage, "Red", &_falbind_GdImage_Red ).asSymbol()
      ->addParam( "color" );
   self->addClassMethod( sym_GdImage, "Rectangle", &_falbind_GdImage_Rectangle ).asSymbol()
      ->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "Line", &_falbind_GdImage_Line ).asSymbol()
      ->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "AlphaBlending", &_falbind_GdImage_AlphaBlending ).asSymbol()
      ->addParam( "alphaBlendingArg" );
   self->addClassMethod( sym_GdImage, "Blue", &_falbind_GdImage_Blue ).asSymbol()
      ->addParam( "color" );
   self->addClassMethod( sym_GdImage, "Char", &_falbind_GdImage_Char ).asSymbol()
      ->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "c" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "DashedLine", &_falbind_GdImage_DashedLine ).asSymbol()
      ->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" )->addParam( "color" );
   self->addExtFunc( "gdImageTrueColor", &_falbind_gdImageTrueColor )
      ->addParam( "im" );
   self->addExtFunc( "gdFontGetLarge", &_falbind_gdFontGetLarge );
   self->addClassMethod( sym_GdImage, "Jpeg", &_falbind_GdImage_Jpeg ).asSymbol()
      ->addParam( "out" )->addParam( "quality" );
   self->addClassMethod( sym_GdImage, "SetBrush", &_falbind_GdImage_SetBrush ).asSymbol()
      ->addParam( "brush" );
   self->addClassMethod( sym_GdImage, "Png", &_falbind_GdImage_Png ).asSymbol()
      ->addParam( "out" );
   self->addClassMethod( sym_GdImage, "ColorClosestHWB", &_falbind_GdImage_ColorClosestHWB ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "GetPixel", &_falbind_GdImage_GetPixel ).asSymbol()
      ->addParam( "x" )->addParam( "y" );
   self->addClassMethod( sym_GdImage, "StringFT", &_falbind_GDImage_stringFT ).asSymbol()
      ->addParam( "fg" )->addParam( "fontname" )->addParam( "ptsize" )->addParam( "angle" )->
      addParam( "x" )->addParam( "y" )->addParam( "string" )->addParam("extra");
   self->addClassMethod( sym_GdImage, "CreateFromGd2", &_falbind_GdImage_CreateFromGd2 ).asSymbol()
      ->addParam( "in" );
   self->addClassMethod( sym_GdImage, "StringUp", &_falbind_GdImage_StringUp ).asSymbol()
      ->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "s" )->addParam( "color" );
   self->addExtFunc( "gdFontGetGiant", &_falbind_gdFontGetGiant );
   self->addClassMethod( sym_GdImage, "CreateFromGd", &_falbind_GdImage_CreateFromGd ).asSymbol()
      ->addParam( "in" );
   self->addClassMethod( sym_GdImage, "ColorResolve", &_falbind_GdImage_ColorResolve ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "SaveAlpha", &_falbind_GdImage_SaveAlpha ).asSymbol()
      ->addParam( "saveAlphaArg" );
   self->addClassMethod( sym_GdImage, "Copy", &_falbind_GdImage_Copy ).asSymbol()
      ->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "w" )->addParam( "h" );
   self->addClassMethod( sym_GdImage, "SetAntiAliasedDontBlend", &_falbind_GdImage_SetAntiAliasedDontBlend ).asSymbol()
      ->addParam( "c" )->addParam( "dont_blend" );
   const Symbol* sym_GdImage_init = self->addExtFunc( "init", &_falbind_GdImage_init, false )
      ->addParam( "sx" )->addParam( "sy" );
   sym_GdImage->getClassDef()->constructor( sym_GdImage_init );
   self->addClassMethod( sym_GdImage, "CopyMergeGray", &_falbind_GdImage_CopyMergeGray ).asSymbol()
      ->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "w" )->addParam( "h" )->addParam( "pct" );
   self->addClassMethod( sym_GdImage, "Fill", &_falbind_GdImage_Fill ).asSymbol()
      ->addParam( "x" )->addParam( "y" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "WBMP", &_falbind_GdImage_WBMP ).asSymbol()
      ->addParam( "fg" )->addParam( "out" );
   self->addClassMethod( sym_GdImage, "GifAnimBegin", &_falbind_GdImage_GifAnimBegin ).asSymbol()
      ->addParam( "out" )->addParam( "GlobalCM" )->addParam( "Loops" );
   self->addClassMethod( sym_GdImage, "ColorClosestAlpha", &_falbind_GdImage_ColorClosestAlpha ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" )->addParam( "a" );
   self->addExtFunc( "gdFontGetTiny", &_falbind_gdFontGetTiny );
   self->addClassMethod( sym_GdImage, "SetClip", &_falbind_GdImage_SetClip ).asSymbol()
      ->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" );
   self->addClassMethod( sym_GdImage, "ColorAllocate", &_falbind_GdImage_ColorAllocate ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "ColorAllocateAlpha", &_falbind_GdImage_ColorAllocateAlpha ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" )->addParam( "a" );
   self->addClassMethod( sym_GdImage, "CopyResized", &_falbind_GdImage_CopyResized ).asSymbol()
      ->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "dstW" )->addParam( "dstH" )->addParam( "srcW" )->addParam( "srcH" );
   self->addClassMethod( sym_GdImage, "CreateFromGd2Part", &_falbind_GdImage_CreateFromGd2Part ).asSymbol()
      ->addParam( "in" )->addParam( "srcx" )->addParam( "srcy" )->addParam( "w" )->addParam( "h" );
   self->addClassMethod( sym_GdImage, "SetStyle", &_falbind_GdImage_SetStyle ).asSymbol()
      ->addParam( "style" )->addParam( "noOfPixels" );
   self->addClassMethod( sym_GdImage, "FilledArc", &_falbind_GdImage_FilledArc ).asSymbol()
      ->addParam( "cx" )->addParam( "cy" )->addParam( "w" )->addParam( "h" )->addParam( "s" )->addParam( "e" )->addParam( "color" )->addParam( "style" );
   self->addExtFunc( "gdFontGetSmall", &_falbind_gdFontGetSmall );
   self->addClassMethod( sym_GdImage, "SetTile", &_falbind_GdImage_SetTile ).asSymbol()
      ->addParam( "tile" );
   self->addClassMethod( sym_GdImage, "PngEx", &_falbind_GdImage_PngEx ).asSymbol()
      ->addParam( "out" )->addParam( "level" );
   self->addClassMethod( sym_GdImage, "GetClip", &_falbind_GdImage_GetClip ).asSymbol()
      ->addParam( "x1P" )->addParam( "y1P" )->addParam( "x2P" )->addParam( "y2P" );
   self->addClassMethod( sym_GdImage, "ColorTransparent", &_falbind_GdImage_ColorTransparent ).asSymbol()
      ->addParam( "color" );
   self->addClassMethod( sym_GdImage, "AABlend", &_falbind_GdImage_AABlend );
   self->addClassMethod( sym_GdImage, "SetAntiAliased", &_falbind_GdImage_SetAntiAliased ).asSymbol()
      ->addParam( "c" );
   self->addClassMethod( sym_GdImage, "Compare", &_falbind_GdImage_Compare ).asSymbol()
      ->addParam( "im2" );
   self->addClassMethod( sym_GdImage, "CharUp", &_falbind_GdImage_CharUp ).asSymbol()
      ->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "c" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "CopyRotated", &_falbind_GdImage_CopyRotated ).asSymbol()
      ->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "srcWidth" )->addParam( "srcHeight" )->addParam( "angle" );
   self->addExtFunc( "gdFontGetMediumBold", &_falbind_gdFontGetMediumBold );
   self->addClassMethod( sym_GdImage, "TrueColorToPalette", &_falbind_GdImage_TrueColorToPalette ).asSymbol()
      ->addParam( "ditherFlag" )->addParam( "colorsWanted" );
   self->addClassMethod( sym_GdImage, "CreateFromJpeg", &_falbind_GdImage_CreateFromJpeg ).asSymbol()
      ->addParam( "infile" );
   self->addClassMethod( sym_GdImage, "CreateFromPng", &_falbind_GdImage_CreateFromPng ).asSymbol()
      ->addParam( "in" );
   self->addClassMethod( sym_GdImage, "FilledRectangle", &_falbind_GdImage_FilledRectangle ).asSymbol()
      ->addParam( "x1" )->addParam( "y1" )->addParam( "x2" )->addParam( "y2" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "SetPixel", &_falbind_GdImage_SetPixel ).asSymbol()
      ->addParam( "x" )->addParam( "y" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "BoundsSafe", &_falbind_GdImage_BoundsSafe ).asSymbol()
      ->addParam( "x" )->addParam( "y" );
   self->addClassMethod( sym_GdImage, "Polygon", &_falbind_GdImage_Polygon ).asSymbol()
      ->addParam( "p" )->addParam( "n" )->addParam( "c" );
   self->addClassMethod( sym_GdImage, "GifAnimEnd", &_falbind_GdImage_GifAnimEnd ).asSymbol()
      ->addParam( "out" );
   self->addClassMethod( sym_GdImage, "Arc", &_falbind_GdImage_Arc ).asSymbol()
      ->addParam( "cx" )->addParam( "cy" )->addParam( "w" )->addParam( "h" )->addParam( "s" )->addParam( "e" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "CopyMerge", &_falbind_GdImage_CopyMerge ).asSymbol()
      ->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "w" )->addParam( "h" )->addParam( "pct" );
   self->addClassMethod( sym_GdImage, "CreateFromWBMP", &_falbind_GdImage_CreateFromWBMP ).asSymbol()
      ->addParam( "infile" );
   self->addClassMethod( sym_GdImage, "GifAnimAdd", &_falbind_GdImage_GifAnimAdd ).asSymbol()
      ->addParam( "out" )->addParam( "LocalCM" )->addParam( "LeftOfs" )->addParam( "TopOfs" )->addParam( "Delay" )->addParam( "Disposal" )->addParam( "previm" );
   self->addClassMethod( sym_GdImage, "SetThickness", &_falbind_GdImage_SetThickness ).asSymbol()
      ->addParam( "thickness" );
   self->addClassMethod( sym_GdImage, "FilledPolygon", &_falbind_GdImage_FilledPolygon ).asSymbol()
      ->addParam( "p" )->addParam( "n" )->addParam( "c" );
   self->addClassMethod( sym_GdImage, "ColorDeallocate", &_falbind_GdImage_ColorDeallocate ).asSymbol()
      ->addParam( "color" );
   self->addClassMethod( sym_GdImage, "Interlace", &_falbind_GdImage_Interlace ).asSymbol()
      ->addParam( "interlaceArg" );
   self->addClassMethod( sym_GdImage, "FilledEllipse", &_falbind_GdImage_FilledEllipse ).asSymbol()
      ->addParam( "cx" )->addParam( "cy" )->addParam( "w" )->addParam( "h" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "CreateTrueColor", &_falbind_GdImage_CreateTrueColor ).asSymbol()
      ->addParam( "sx" )->addParam( "sy" );
   self->addClassMethod( sym_GdImage, "FillToBorder", &_falbind_GdImage_FillToBorder ).asSymbol()
      ->addParam( "x" )->addParam( "y" )->addParam( "border" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "OpenPolygon", &_falbind_GdImage_OpenPolygon ).asSymbol()
      ->addParam( "p" )->addParam( "n" )->addParam( "c" );
   self->addClassMethod( sym_GdImage, "SX", &_falbind_GdImage_SX );
   self->addClassMethod( sym_GdImage, "SY", &_falbind_GdImage_SY );
   self->addClassMethod( sym_GdImage, "ColorClosest", &_falbind_GdImage_ColorClosest ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "CreatePaletteFromTrueColor", &_falbind_GdImage_CreatePaletteFromTrueColor ).asSymbol()
      ->addParam( "ditherFlag" )->addParam( "colorsWanted" );
   self->addClassMethod( sym_GdImage, "ColorResolveAlpha", &_falbind_GdImage_ColorResolveAlpha ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" )->addParam( "a" );
   self->addClassMethod( sym_GdImage, "String", &_falbind_GdImage_String ).asSymbol()
      ->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "s" )->addParam( "color" );
   self->addClassMethod( sym_GdImage, "Gif", &_falbind_GdImage_Gif ).asSymbol()
      ->addParam( "out" );
   self->addClassMethod( sym_GdImage, "ColorExactAlpha", &_falbind_GdImage_ColorExactAlpha ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" )->addParam( "a" );
   self->addClassMethod( sym_GdImage, "CopyResampled", &_falbind_GdImage_CopyResampled ).asSymbol()
      ->addParam( "src" )->addParam( "dstX" )->addParam( "dstY" )->addParam( "srcX" )->addParam( "srcY" )->addParam( "dstW" )->addParam( "dstH" )->addParam( "srcW" )->addParam( "srcH" );
   self->addClassMethod( sym_GdImage, "Alpha", &_falbind_GdImage_Alpha ).asSymbol()
      ->addParam( "color" );
   self->addClassMethod( sym_GdImage, "CreateFromGif", &_falbind_GdImage_CreateFromGif ).asSymbol()
      ->addParam( "in" );
   self->addClassMethod( sym_GdImage, "GetTrueColorPixel", &_falbind_GdImage_GetTrueColorPixel ).asSymbol()
      ->addParam( "x" )->addParam( "y" );
   self->addClassMethod( sym_GdImage, "ColorExact", &_falbind_GdImage_ColorExact ).asSymbol()
      ->addParam( "r" )->addParam( "g" )->addParam( "b" );
   self->addClassMethod( sym_GdImage, "Green", &_falbind_GdImage_Green ).asSymbol()
      ->addParam( "color" );
   self->addClassMethod( sym_GdImage, "StringUp16", &_falbind_GdImage_StringUp16 ).asSymbol()
      ->addParam( "f" )->addParam( "x" )->addParam( "y" )->addParam( "s" )->addParam( "color" );
   return self;
}

/* Done with falbind */

