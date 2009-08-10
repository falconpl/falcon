/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_ext.cpp

   The SDL True Type binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:11:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL True Type binding support module.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>

#include "sdlttf_ext.h"
#include "sdlttf_mod.h"
#include <sdl_service.h>  // for the reset
extern "C"
{
   #include <SDL_ttf.h>
}

/*# @beginmodule sdlttf */

namespace Falcon {

static SDLService *s_service = 0;

namespace Ext {

/*#
   @method Init TTF
   @brief Initialize the TTF module
   @raise SDLError on initialization failure

   Does not require @a SDL.Init to be called before.
*/

FALCON_FUNC ttf_Init( VMachine *vm )
{
   int retval = ::TTF_Init();
   if ( retval < 0 )
   {
      throw new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE, __LINE__ )
         .desc( "TTF Error" )
         .extra( TTF_GetError() ) ) ;
      return;
   }

   // we can be reasonabily certain that our service is ready here.
   s_service = (SDLService *) vm->getService( "SDLService" );
   if ( s_service == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE+2, __LINE__ )
         .desc( "SDL service not in the target VM" ) ) ;
   }
}

/*#
   @method WasInit TTF
   @brief Detects if the TTF subsystem was initialized.
   @return True if the system was initialized, false otherwise.
*/

FALCON_FUNC ttf_WasInit( VMachine *vm )
{
   vm->retval( (TTF_WasInit() ? true : false ) );
}


/*#
   @method InitAuto TTF
   @brief Initialize the TTF module and prepares for automatic de-initialization.
   @raise SDLError on initialization failure

   Does not require @a SDL.Init to be called before.

   This method returns an object; when the object is destroyed by the GC,
   the library is de-initialized. To perform an application wide intialization
   with automatic de-initialization on quit, just store the return value of
   this method on a global variable of the main script.
*/

FALCON_FUNC ttf_InitAuto( VMachine *vm )
{
   int retval = ::TTF_Init();
   if ( retval < 0 )
   {
      throw new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE, __LINE__ )
         .desc( "TTF Init error" )
         .extra( TTF_GetError() ) ) ;
      return;
   }

   // we can be reasonabily certain that our service is ready here.
   s_service = (SDLService *) vm->getService( "SDLService" );
   if ( s_service == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE+2, __LINE__ )
         .desc( "SDL service not in the target VM" ) ) ;
   }

   // also create an object for auto quit.
   Item *c_auto = vm->findWKI( "_TTF_AutoQuit" );
   fassert( c_auto != 0 );
   CoreObject *obj = c_auto->asClass()->createInstance();
   obj->setUserData( new TTFQuitCarrier );
   vm->retval( obj );

}

/*#
   @method Quit TTF
   @brief Turns off TTF system.

   This call shuts down the TTF extensions for SDL and resets system status.
   After a TTF quit, it is possible to reinitialize it.
*/

FALCON_FUNC ttf_Quit( VMachine *vm )
{
   ::TTF_Quit();
}

/*#
   @method Compiled_Version TTF
   @brief Determine the version used to compile this SDL TTF module.
   @return a three element array containing the major, minor and fix versions.
   @see TTF.Linked_Version
*/
FALCON_FUNC ttf_Compiled_Version( VMachine *vm )
{
   SDL_version compile_version;
   TTF_VERSION(&compile_version);

   CoreArray *arr = new CoreArray( 3 );
   arr->append( (int64) compile_version.major );
   arr->append( (int64) compile_version.minor );
   arr->append( (int64) compile_version.patch );
   vm->retval( arr );
}

/*#
   @method Linked_Version TTF
   @brief Determine the version of the library that is currently linked.
   @return a three element array containing the major, minor and fix versions.

   This function determines the version of the SDL_ttf library that is running
   on the system. As long as the interface is the same, it may be different
   from the version used to compile this module.
*/
FALCON_FUNC ttf_Linked_Version( VMachine *vm )
{
   const SDL_version *link_version;
   link_version = TTF_Linked_Version();

   CoreArray *arr = new CoreArray( 3 );
   arr->append( (int64) link_version->major );
   arr->append( (int64) link_version->minor );
   arr->append( (int64) link_version->patch );
   vm->retval( arr );
}


/*#
   @method ByteSwappedUNICODE TTF
   @brief Set default UNICODE byte swapping mode.
   @param swap Swapping mode.

   If swap is non-zero then UNICODE data is byte swapped relative to the CPU's native endianness.
   if swap zero, then do not swap UNICODE data, use the CPU's native endianness.
*/
FALCON_FUNC ttf_ByteSwappedUNICODE( VMachine *vm )
{
   Item *i_mode = vm->param(0);

   if( i_mode == 0 || ! i_mode->isOrdinal() )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) ;
      return;
   }

   ::TTF_ByteSwappedUNICODE( (int) i_mode->forceInteger() );
}

/*#
   @method OpenFont TTF
   @brief Open a font file.
   @param fontname File where the font is stored.
   @param ptsize Size of the font that must be loaded.
   @optparam index Face index in the file.

   @return An instance of class @a TTFFont.
   @raise SDLError if the font cannot be loaded.
*/
FALCON_FUNC ttf_OpenFont( VMachine *vm )
{
   Item *i_filename = vm->param(0);
   Item *i_ptsize = vm->param(1);
   Item *i_index = vm->param(2);

   if( i_filename == 0 || ! i_filename->isString() ||
       i_ptsize == 0 || ! i_ptsize->isOrdinal() ||
       ( i_index != 0 && ! i_index->isOrdinal() )
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S,N,[N]" ) ) ;
      return;
   }

   long index = i_index != 0 ? ((long)i_index->forceInteger()) : 0;
   AutoCString file( *i_filename->asString() );

   ::TTF_Font *fnt = ::TTF_OpenFontIndex( file.c_str(),
         (int) i_ptsize->forceInteger(), index );

   if( fnt == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE + 1, __LINE__ )
         .desc( "TTF Load error" )
         .extra( TTF_GetError() ) ) ;
      return;
   }

   Item *c_font = vm->findWKI( "TTFFont" );
   CoreObject *obj = c_font->asClass()->createInstance();
   obj->setUserData( new TTFFontCarrier( fnt ) );
   vm->retval( obj );
}

/*#
   @method GetFontStyle TTFFont
   @brief Get font render style
   @return Style currently active on this font.

   Returned style can be TTF.STYLE_NORMAL or a bitfield or'd combination of the following:

   - TTF.STYLE_BOLD
   - TTF.STYLE_ITALIC
   - TTF.STYLE_UNDERLINE
*/
FALCON_FUNC ttf_GetFontStyle( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   vm->retval( (int64) ::TTF_GetFontStyle( font ) );
}

/*#
   @method SetFontStyle TTFFont
   @brief Set the rendering style of the loaded font.
   @param style Render style of this font.

   The parameter can be TTF.STYLE_NORMAL or a bitfield or'd combination of the following:

   - TTF.STYLE_BOLD
   - TTF.STYLE_ITALIC
   - TTF.STYLE_UNDERLINE
*/
FALCON_FUNC ttf_SetFontStyle( VMachine *vm )
{
   Item *i_style = vm->param( 0 );

   if( i_style == 0 || ! i_style->isOrdinal() )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) ;
      return;
   }

   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   ::TTF_SetFontStyle( font, (int) i_style->forceInteger() );
}

/*#
   @method FontHeight TTFFont
   @brief Get the maximum pixel height of all glyphs of the loaded font.
   @return Height of the font, in pixels.

   You may use this height for rendering text as close together vertically as possible,
   though adding at least one pixel height to it will space it so they can't touch.
   Remember that SDL_ttf doesn't handle multiline printing, so you are responsible for
   line spacing, see the TTF_FontLineSkip as well.
*/

FALCON_FUNC ttf_FontHeight( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   vm->retval( (int64) ::TTF_FontHeight( font ) );
}


/*#
   @method FontAscent TTFFont
   @brief Get the maximum pixel ascent of all glyphs of the loaded font.
   @return Height of the ascent, in pixels.
*/
FALCON_FUNC ttf_FontAscent( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   vm->retval( (int64) ::TTF_FontAscent( font ) );
}

/*#
   @method FontDescent TTFFont
   @brief Get the maximum pixel descent of all glyphs of the loaded font.
   @return Height of the descent, in pixels.
*/
FALCON_FUNC ttf_FontDescent( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   vm->retval( (int64) ::TTF_FontDescent( font ) );
}


/*#
   @method FontLineSkip TTFFont
   @brief Get the recommended pixel height of a rendered line of text of the loaded font.
   @return Recommended height, in pixels.
*/
FALCON_FUNC ttf_FontLineSkip( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   vm->retval( (int64) ::TTF_FontLineSkip( font ) );
}


/*#
   @method FontFaces TTFFont
   @brief Get the number of faces ("sub-fonts") available in the loaded font.
   @return Number of faces stored in this font.
*/
FALCON_FUNC ttf_FontFaces( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   vm->retval( (int64) ::TTF_FontFaces( font ) );
}

/*#
   @method FontFaceIsFixedWidth TTFFont
   @brief Determines wether this font has fixed width or not.
   @return True if the font is fixed width, false otherise.
*/
FALCON_FUNC ttf_FontFaceIsFixedWidth( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   vm->retval( (bool) (::TTF_FontFaces( font ) > 0) );
}

/*#
   @method FontFaceFamilyName TTFFont
   @brief Determine the family name of this font.
   @return A string containing the family name of this font.
*/
FALCON_FUNC ttf_FontFaceFamilyName( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   const char *family = ::TTF_FontFaceFamilyName( font );
   vm->retval( new CoreString( family ) );
}

/*#
   @method FontFaceStyleName TTFFont
   @brief Determine the face style name of this font.
   @return A string containing the style name, or @b nil if not available.
*/
FALCON_FUNC ttf_FontFaceStyleName( VMachine *vm )
{
   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   const char *style = ::TTF_FontFaceStyleName( font );
   if ( style != 0 )
      vm->retval( new CoreString( style ) );
   else
      vm->retnil();
}

/*#
   @method GlyphMetrics TTFFont
   @brief Returns the metrics of a determined gliph in this font.
   @param charId the ID of the caracter of which the gliph are required.
   @optparam metrics An instance of class @a TTFMetrics (or providing the same properties).
   @return An instance of class @a TTFMetrics or nil.

   This method stores the glyph metrics of character determined by the given @b charId into
   the @b minx, @b maxx, @b miny, @b maxy and @b advance properties of the object given
   as second parameter. If that object is not given, this method creates an instance of the
   @a TTFMetrics class and returns it. It is advisable to use repeatedly such instance
   to avoid useless garbage generation.

   The function may fail if the @b charId is not found.

   IF a fitting instace is provided as a parameter, then that value is returned. On failure,
   nil is returned.

   @note as object are always evaluated to true, the fact that this function returns an object
   can be used to determine if it was succesful or not.
*/
FALCON_FUNC ttf_GlyphMetrics( VMachine *vm )
{
   Item *i_charId = vm->param( 0 );
   Item *i_storage = vm->param( 1 );

   if( i_charId == 0 || ! i_charId->isOrdinal() ||
       ( i_storage != 0 && !i_storage->isObject() )
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,[O]" ) ) ;
      return;
   }

   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   Uint16 charId = (Uint16) i_charId->forceInteger();

   int minx, maxx, miny, maxy, advance;
   int res = ::TTF_GlyphMetrics( font, charId, &minx, &maxx, &miny, &maxy, &advance);
   if ( res < 0 )
   {
      vm->retnil();
      return;
   }

   CoreObject *metrics;
   if( i_storage != 0 )
      metrics = i_storage->asObject();
   else
   {
      Item *c_metrics = vm->findWKI( "TTFMetrics" );
      fassert( c_metrics != 0 );
      metrics = c_metrics->asClass()->createInstance();
   }

   metrics->setProperty( "minx", (int64) minx );
   metrics->setProperty( "maxx", (int64) maxx );
   metrics->setProperty( "miny", (int64) miny );
   metrics->setProperty( "maxy", (int64) maxy );
   metrics->setProperty( "advance", (int64) advance );
   vm->retval( metrics );
}

/*#
   @method SizeText TTFFont
   @brief Determine rendering size of a text as it would be rendere on ttf.
   @param string A string to be rendered.
   @optparam metrics An instance of class @a TTFMetrics (or providing the same properties).
   @return An instance of class @a TTFMetrics.

   This method stores the width and height of the rendered string on the properties @b w and
   @b h of a given object.

   The function may fail if the some of the characters in the string cannot be represented
   with by this font.

   @note This function translates the input string in an UTF-8 string and then feeds it into
   TTF_SizeUTF8 as this is guaranteed to work with every character known by falcon and
   across any platform.

   IF a fitting instace is provided as a parameter, then that value is returned. On failure,
   nil is returned.

   @note as object are always evaluated to true, the fact that this function returns an object
   can be used to determine if it was succesful or not.
*/
FALCON_FUNC ttf_SizeText( VMachine *vm )
{
   Item *i_string = vm->param( 0 );
   Item *i_storage = vm->param( 1 );

   if( i_string == 0 || ! i_string->isString() ||
       ( i_storage != 0 && !i_storage->isObject() )
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S,[O]" ) ) ;
      return;
   }

   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;
   AutoCString utf8_str( *i_string->asString() );

   int w, h;
   int res = ::TTF_SizeUTF8( font, utf8_str.c_str(), &w, &h );
   if ( res < 0 )
   {
      vm->retnil();
      return;
   }

   CoreObject *metrics;
   if( i_storage != 0 )
      metrics = i_storage->asObject();
   else
   {
      Item *c_metrics = vm->findWKI( "TTFMetrics" );
      fassert( c_metrics != 0 );
      metrics = c_metrics->asClass()->createInstance();
   }

   metrics->setProperty( "w", (int64) w );
   metrics->setProperty( "h", (int64) h );
   vm->retval( metrics );
}


static bool internal_object_to_color( CoreObject *obj_color, SDL_Color &color )
{
   Item prop;
   if( obj_color->getProperty( "r", prop ) )
   {
      color.r = (Uint8) prop.forceInteger();
      if( obj_color->getProperty( "g", prop ) )
      {
         color.g = (Uint8) prop.forceInteger();
         if( obj_color->getProperty( "b", prop ) )
         {
            color.b = (Uint8) prop.forceInteger();
            return true;
         }
      }
   }

   return false;
}

static void internal_render( VMachine *vm, int mode )
{
   Item *i_string = vm->param( 0 );
   Item *i_color = vm->param( 1 );
   Item *i_colorbg = vm->param( 2 );

   if( i_string == 0 || ( ! i_string->isString() && ! i_string->isOrdinal() ) ||
       i_color == 0 && !i_color->isObject() ||
       (mode == 1 && (i_colorbg == 0 || ! i_colorbg->isObject() ))
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N|S,O" ) ) ;
      return;
   }

   // extract the color -- if possible
   ::SDL_Color color;
   ::SDL_Color colorbg;

   // Ops... object was wrong ?
   if ( ! internal_object_to_color(i_color->asObject(), color ) ||
        ( mode == 1 && ! internal_object_to_color(i_colorbg->asObject(), colorbg ) )
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Object is not a color" ) ) ;
      return;
   }
   // we need the service here
   if ( s_service == 0 )
   {
       throw new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE+2, __LINE__ )
         .desc( "Service not initialized" ) ) ;
   }

   ::TTF_Font *font = static_cast<TTFFontCarrier *>(vm->self().asObject()->getUserData())->m_font;

   SDL_Surface *text_surface = 0;
   // if the thing is a string...
   if( i_string->isString() )
   {
      AutoCString utf8_str( *i_string->asString() );
      switch( mode )
      {
         case 0:
            text_surface = ::TTF_RenderUTF8_Solid( font, utf8_str.c_str(), color );
            break;
         case 1:
            text_surface = ::TTF_RenderUTF8_Shaded( font, utf8_str.c_str(), color, colorbg );
            break;
         case 2:
            text_surface = ::TTF_RenderUTF8_Blended( font, utf8_str.c_str(), color );
            break;
      }
   }
   else {
      switch( mode )
      {
         case 0:
            text_surface = ::TTF_RenderGlyph_Solid( font, (Uint8) i_string->forceInteger(), color );
            break;
         case 1:
            text_surface = ::TTF_RenderGlyph_Shaded( font, (Uint8) i_string->forceInteger(), color, colorbg );
            break;
         case 2:
            text_surface = ::TTF_RenderGlyph_Blended( font, (Uint8) i_string->forceInteger(), color );
            break;
      }
   }

   if ( text_surface == 0 )
   {
       throw new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE+1, __LINE__ )
         .desc( "TTF Render Error" )
         .extra( TTF_GetError() ) ) ;
   }

   // we need the service here
   vm->retval( s_service->createSurfaceInstance(vm, text_surface ) );
}

/*#
   @method Render_Solid TTFFont
   @brief Renders text on a final SDLSurface instance (fast).
   @param string A string or a character to be rendered.
   @param color an @a SDLColor instance containing the color of the string.
   @return an @a SDLSurface containing the rendered string.
   @raise SDLError on failure.

   The first parameter may be a string or a numeric value, which will be interpreted
   as a single glyph to be rendered.

   The @b color parameter may be any object providing @b r, @b g and @b b properties.

   The function may fail if the some of the characters in the string cannot be represented
   with by this font.

   @note This function translates the input string in an UTF-8 string and then feeds it into
   TTF_RenderUTF_Solid as this is guaranteed to work with every character known by falcon and
   across any platform.
*/
FALCON_FUNC ttf_Render_Solid( VMachine *vm )
{
   internal_render( vm, 0 );
}

/*#
   @method Render_Shaded TTFFont
   @brief Renders text on a final SDLSurface instance (medium).
   @param string A string or a character to be rendered.
   @param color an @a SDLColor instance containing the color of the string.
   @param bgcolor an @a SDLColor instance containing the color of the background.
   @return an @a SDLSurface containing the rendered string.
   @raise SDLError on failure.

   The first parameter may be a string or a numeric value, which will be interpreted
   as a single glyph to be rendered.

   The @b color parameter may be any object providing @b r, @b g and @b b properties.

   The function may fail if the some of the characters in the string cannot be represented
   with by this font.

   @note This function translates the input string in an UTF-8 string and then feeds it into
   TTF_RenderUTF_Solid as this is guaranteed to work with every character known by falcon and
   across any platform.
*/
FALCON_FUNC ttf_Render_Shaded( VMachine *vm )
{
   internal_render( vm, 1 );
}

/*#
   @method Render_Blended TTFFont
   @brief Renders text on a final SDLSurface instance (medium).
   @param string A string or a character to be rendered.
   @param color an @a SDLColor instance containing the color of the string.
   @return an @a SDLSurface containing the rendered string.
   @raise SDLError on failure.

   The first parameter may be a string or a numeric value, which will be interpreted
   as a single glyph to be rendered.

   The @b color parameter may be any object providing @b r, @b g and @b b properties.

   The function may fail if the some of the characters in the string cannot be represented
   with by this font.

   @note This function translates the input string in an UTF-8 string and then feeds it into
   TTF_RenderUTF_Solid as this is guaranteed to work with every character known by falcon and
   across any platform.
*/
FALCON_FUNC ttf_Render_Blended( VMachine *vm )
{
   internal_render( vm, 2 );
}

}
}

/* end of TTF_ext.cpp */
