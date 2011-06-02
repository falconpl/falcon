/**
 *  \file gdk_Pixbuf.cpp
 */

#include "gdk_Pixbuf.hpp"
#include <falcon/path.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Pixbuf::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Pixbuf = mod->addClass( "GdkPixbuf", &Pixbuf::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_Pixbuf->getClassDef()->addInheritance( in );

    c_Pixbuf->setWKS( true );
    c_Pixbuf->getClassDef()->factory( &Pixbuf::factory );

    Gtk::MethodTab methods[] =
    {
    { "version",				&Pixbuf::version },
    { "get_n_channels",			&Pixbuf::get_n_channels },
    { "get_has_alpha",			&Pixbuf::get_has_alpha },
    { "get_bits_per_sample",    &Pixbuf::get_bits_per_sample },
    { "get_pixels",				&Pixbuf::get_pixels },
    { "get_width",				&Pixbuf::get_width },
    { "get_height",				&Pixbuf::get_height },
    { "new_from_file",			&Pixbuf::new_from_file },
    { "new_from_file_at_size",  &Pixbuf::new_from_file_at_size },
    { "new_from_file_at_scale", &Pixbuf::new_from_file_at_scale },
    { "flip",					&Pixbuf::flip },
    { "rotate_simple",			&Pixbuf::rotate_simple },
	{ "scale_simple",			&Pixbuf::scale_simple },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Pixbuf, meth->name, meth->cb );
}


Pixbuf::Pixbuf( const Falcon::CoreClass* gen, const GdkPixbuf* buf )
    :
    Gtk::CoreGObject( gen, (GObject*) buf )
{}


Falcon::CoreObject* Pixbuf::factory( const Falcon::CoreClass* gen, void* buf, bool )
{
    return new Pixbuf( gen, (GdkPixbuf*) buf );
}

/*#
  @class GdkPixbuf
  @brief
  @param
  @param
  @param
  @param

  Some description here
*/

FALCON_FUNC Pixbuf::init( VMARG )
{
  Item* i_colorspace = vm->param( 0 );
  Item* i_has_alpha = vm->param( 1 );
  Item* i_bits_per_sample = vm->param( 2 );
  Item* i_width = vm->param( 3 );
  Item* i_height = vm->param( 4 );

#ifndef NO_PARAMETER_CHECK
  if ( !i_colorspace || !i_colorspace->isInteger()
       || !i_has_alpha || !i_has_alpha->isTrue()
       || !i_bits_per_sample || !i_bits_per_sample->isInteger()
       || !i_width || !i_width->isInteger()
       || !i_height || !i_height->isInteger() )
    throw_inv_params( "I,I,I,I,I" );
#endif
  MYSELF;
  self->setObject( (GObject*) gdk_pixbuf_new( (GdkColorspace) i_colorspace->asInteger(),
					      i_has_alpha->isTrue(),
					      i_bits_per_sample->asInteger(),
					      i_width->asInteger(),
					      i_height->asInteger() ) );
}



/*#
    @class GdkPixbuf
    @brief Information that describes an image.
 */

/*#
    @method version GdkPixbuf
    @brief a static function returning the GdkPixbuf version.
    @return the full version of the gdk-pixbuf library as a string.

    This is the version currently in use by a running program.
 */
FALCON_FUNC Pixbuf::version( VMARG )
{
  NO_ARGS
  vm->retval( UTF8String( GDK_PIXBUF_VERSION ) );
}

/*#

*/
FALCON_FUNC Pixbuf::get_n_channels( VMARG )
{
  NO_ARGS

  MYSELF;
  GET_OBJ( self );

  vm->retval( (int64) gdk_pixbuf_get_n_channels( GET_PIXBUF( vm->self() ) ) );
}

/*#

*/
FALCON_FUNC Pixbuf::get_has_alpha( VMARG )
{
  NO_ARGS

  MYSELF;
  GET_OBJ( self );

  vm->retval( (bool) gdk_pixbuf_get_has_alpha( GET_PIXBUF( vm->self() ) ) );
}

/*#

*/
FALCON_FUNC Pixbuf::get_bits_per_sample( VMARG )
{
  NO_ARGS

  MYSELF;
  GET_OBJ( self );

  vm->retval( (int64) gdk_pixbuf_get_bits_per_sample( GET_PIXBUF( vm->self() ) ) );
}

/*#

*/
FALCON_FUNC Pixbuf::get_pixels( VMARG )
{
  NO_ARGS

  MYSELF;
  GET_OBJ( self );

  guchar* pixels = gdk_pixbuf_get_pixels( GET_PIXBUF( vm->self() ) );

  //  vm->retval( UTF8String( gdk_pixbuf_get_pixels( GET_PIXBUF( vm->self() ) ) ) );
  vm->retval( pixels ); // UTF8String ??
}


/*#

*/
FALCON_FUNC Pixbuf::get_width( VMARG )
{
  NO_ARGS

  MYSELF;
  GET_OBJ( self );

  vm->retval( (int64) gdk_pixbuf_get_width( GET_PIXBUF( vm->self() ) ) );
}

/*#

*/
FALCON_FUNC Pixbuf::get_height( VMARG )
{
  NO_ARGS
  vm->retval( (int64) gdk_pixbuf_get_height( GET_PIXBUF( vm->self() ) ) );
}


/*#

*/
FALCON_FUNC Pixbuf::new_from_file( VMARG )
{
  Item* i_filename = vm->param( 0 );

#ifndef NO_PARAMETER_CHECK
  if ( !i_filename || !i_filename->isString() )
    throw_inv_params( "[S]" );
#endif

  String* filename = i_filename->asString();

  Path path( *filename );

  // Following is taken from GtkImage
#ifdef FALCON_SYSTEM_WIN
  filename->size( 0 );
  path.getWinFormat( *filename );
#else
  filename->copy( path.get() );
#endif
  //

  AutoCString s( filename );

  GError* err = NULL;

  GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file( s.c_str(), &err );

  if ( err != NULL )
    {
      g_print( err->message );
      g_error_free( err  );
    }

  vm->retval( new Pixbuf( vm->findWKI( "GdkPixbuf" )->asClass(), (GdkPixbuf*) pixbuf ) );
}


/*#

*/
FALCON_FUNC Pixbuf::new_from_file_at_size( VMARG )
{
  Item* i_filename = vm->param( 0 );
  Item* i_width = vm->param( 1 );
  Item* i_height = vm->param( 2 );

#ifndef NO_PARAMETER_CHECK
  if ( !i_filename || !i_filename->isString()
       || !i_width || !i_width->isInteger()
       || !i_height || !i_height->isInteger() )
    throw_inv_params( "S,I,I" );
#endif

  String* filename = i_filename->asString();
  Path path( *filename );

  // Following is taken from GtkImage
#ifdef FALCON_SYSTEM_WIN
  filename->size( 0 );
  path.getWinFormat( *filename );
#else
  filename->copy( path.get() );
#endif
  //

  AutoCString s( filename );

  GError* err = NULL;

  GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file_at_size( s.c_str(), i_width->asInteger(), i_height->asInteger(), &err );

  if ( err != NULL )
    {
      g_print( err->message );
      g_error_free( err  );
    }

  vm->retval( new Gdk::Pixbuf( vm->findWKI( "GdkPixbuf" )->asClass(), (GdkPixbuf*) pixbuf ) );
}

/*#

*/
FALCON_FUNC Pixbuf::new_from_file_at_scale( VMARG )
{
  Item* i_filename = vm->param( 0 );
  Item* i_width = vm->param( 1 );
  Item* i_height = vm->param( 2 );
  Item* i_preserve_aspect_ratio = vm->param( 3 );


#ifndef NO_PARAMETER_CHECK
  if ( !i_filename || !i_filename->isString()
       || !i_width || !i_width->isInteger()
       || !i_height || !i_height->isInteger()
       || !i_preserve_aspect_ratio || !i_preserve_aspect_ratio->isTrue() )
    throw_inv_params( "S,I,I,B" );
#endif

  String* filename = i_filename->asString();

  Path path( *filename );

  // Following is taken from GtkImage
#ifdef FALCON_SYSTEM_WIN
  filename->size( 0 );
  path.getWinFormat( *filename );
#else
  filename->copy( path.get() );
#endif
  //

  AutoCString s( filename );

  GError* err = NULL;

  GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file_at_scale( s.c_str(), i_width->asInteger(), i_height->asInteger(), (gboolean) i_preserve_aspect_ratio->isTrue(), &err );

  if ( err != NULL )
    {
      g_print( err->message );
      g_error_free( err  );
    }

  vm->retval( new Gdk::Pixbuf( vm->findWKI( "GdkPixbuf" )->asClass(), (GdkPixbuf*) pixbuf ) );
}


/*#

*/
FALCON_FUNC Pixbuf::flip( VMARG )
{
  Item* i_horizontal = vm->param( 0 );

  MYSELF;
  GET_OBJ( self );

#ifndef NO_PARAMETER_CHECK
  if ( !(i_horizontal || i_horizontal->isTrue() ) )
    throw_inv_params( "[B]" );
#endif

  vm->retval( new Gdk::Pixbuf( vm->findWKI( "GdkPixbuf" )->asClass(),
			       gdk_pixbuf_flip( GET_PIXBUF( vm->self() ),
						i_horizontal->isTrue() ) ) );
}



/*#

*/
FALCON_FUNC Pixbuf::rotate_simple( VMARG )
{
  Item* i_angle = vm->param( 0 );

  MYSELF;
  GET_OBJ( self );

#ifndef NO_PARAMETER_CHECK
  if ( !( i_angle || i_angle->isInteger() ) )
    throw_inv_params( "[I]" );
#endif
  g_print("123");
  vm->retval( new Gdk::Pixbuf( vm->findWKI( "GdkPixbuf" )->asClass(),
			       gdk_pixbuf_rotate_simple( GET_PIXBUF( vm->self() ) ,
							 (GdkPixbufRotation) i_angle->asInteger() ) ) );
}

/*#
    @method version scale_simple
    @brief a static function returning the GdkPixbuf version.
    @return the full version of the gdk-pixbuf library as a string.

    This is the version currently in use by a running program.
 */
FALCON_FUNC Pixbuf::scale_simple( VMARG )
{
  Item* i_dest_width  = vm->param( 0 );
  Item* i_dest_height = vm->param( 1 );
  Item* i_interp_type = vm->param( 2 );

#ifndef NO_PARAMETER_CHECK
    if ( !i_dest_width || !i_dest_width->isInteger()
		|| !i_dest_height || !i_dest_height->isInteger() 
		|| !i_interp_type || !i_interp_type->isInteger() 
	   )
		throw_inv_params( "[I,I,I]" );
#endif
	MYSELF;

	self->setObject( (GObject*) gdk_pixbuf_scale_simple( GET_PIXBUF( vm->self() ), 
															 i_dest_width->asInteger(),
															 i_dest_height->asInteger(), 
															 (GdkInterpType) i_interp_type->asInteger() ) );
					
}

} // Gdk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
