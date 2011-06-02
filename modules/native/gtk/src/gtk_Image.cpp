/**
 *  \file gtk_Image.cpp
 */

#include "gtk_Image.hpp"
#include "gdk_Pixbuf.hpp"

#include <gtk/gtk.h>

#include <falcon/path.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Image::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Image = mod->addClass( "GtkImage", &Image::init );

    c_Image->setWKS( true );
    c_Image->getClassDef()->factory( &Image::factory );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkMisc" ) );
    c_Image->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    //{ "get_icon_set",        &Image::foo },
    //{ "get_image",        &Image::foo },
    //{ "get_pixbuf",        &Image::foo },
    //{ "get_pixmap",        &Image::foo },
    //{ "get_stock",        &Image::foo },
    //{ "get_animation",        &Image::foo },
    //{ "get_icon_name",        &Image::foo },
    //{ "get_gicon",        &Image::foo },
    //{ "get_storage_type",        &Image::foo },
    //{ "new_from_file",        &Image::foo },
    //{ "new_from_icon_set",        &Image::foo },
    //{ "new_from_image",        &Image::foo },
    { "new_from_pixbuf",        &Image::new_from_pixbuf },
    //{ "new_from_pixmap",        &Image::foo },
    { "new_from_stock",         &Image::new_from_stock },
    //{ "new_from_animation",        &Image::foo },
    //{ "new_from_icon_name",        &Image::foo },
    //{ "new_from_gicon",        &Image::foo },
    { "set_from_file",          &Image::set_from_file },
    //{ "set_from_icon_set",        &Image::foo },
    //{ "set_from_image",        &Image::foo },
    { "set_from_pixbuf",        &Image::set_from_pixbuf },
    //{ "set_from_pixmap",        &Image::foo },
    { "set_from_stock",         &Image::set_from_stock },
    //{ "set_from_animation",        &Image::foo },
    //{ "set_from_icon_name",        &Image::foo },
    //{ "set_from_gicon",        &Image::foo },
    { "clear",                  &Image::clear },
    //{ "set",        &Image::foo },
    //{ "get",        &Image::foo },
    //{ "set_pixel_size",        &Image::foo },
    //{ "get_pixel_size",        &Image::foo },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Image, meth->name, meth->cb );
}


Image::Image( const Falcon::CoreClass* gen, const GtkImage* img )
    :
    Gtk::CoreGObject( gen, (GObject*) img )
{}


Falcon::CoreObject* Image::factory( const Falcon::CoreClass* gen, void* img, bool )
{
    return new Image( gen, (GtkImage*) img );
}


/*#
    @class GtkImage
    @brief A widget displaying an image
    @optparam filename a filename (string)

    The GtkImage widget displays an image. Various kinds of object can be displayed
    as an image; most typically, you would load a GdkPixbuf ("pixel buffer") from a
    file, and then display that.

    [...]

    GtkImage is a subclass of GtkMisc, which implies that you can align it (center,
    left, right) and add padding to it, using GtkMisc methods.

    GtkImage is a "no window" widget (has no GdkWindow of its own), so by default
    does not receive events. If you want to receive events on the image, such as
    button clicks, place the image inside a GtkEventBox, then connect to the event
    signals on the event box.
 */
FALCON_FUNC Image::init( VMARG )
{
    Item* i_fnam = vm->param( 0 );
    GtkWidget* img;
    if ( i_fnam )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_fnam->isNil() || !i_fnam->isString() )
            throw_inv_params( "[S]" );
#endif
        AutoCString s( i_fnam->asString() );
        img = gtk_image_new_from_file( s.c_str() );
    }
    else
        img = gtk_image_new();

    MYSELF;
    self->setObject( (GObject*) img );
}


//FALCON_FUNC Image::get_icon_set( VMARG );

//FALCON_FUNC Image::get_image( VMARG );

//FALCON_FUNC Image::get_pixbuf( VMARG );

//FALCON_FUNC Image::get_pixmap( VMARG );

//FALCON_FUNC Image::get_stock( VMARG );

//FALCON_FUNC Image::get_animation( VMARG );

//FALCON_FUNC Image::get_icon_name( VMARG );

//FALCON_FUNC Image::get_gicon( VMARG );

//FALCON_FUNC Image::get_storage_type( VMARG );

//FALCON_FUNC Image::new_from_file( VMARG );

//FALCON_FUNC Image::new_from_icon_set( VMARG );

//FALCON_FUNC Image::new_from_image( VMARG );

/*#
 *
 */

FALCON_FUNC Image::new_from_pixbuf( VMARG )
{
  Item* i_pixbuf = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
  if ( !i_pixbuf || !( i_pixbuf->isNil() || ( i_pixbuf->isObject() && IS_DERIVED( i_pixbuf, GdkPixbuf ) ) ) )
    throw_inv_params( "[GdkPixbuf]" );
#endif

  GtkWidget* img = gtk_image_new_from_pixbuf( GET_PIXBUF( *i_pixbuf ) );

  vm->retval( new Image( vm->self().asClass(), (GtkImage*) img ) );
}

//FALCON_FUNC Image::new_from_pixmap( VMARG );


/*#
    @method new_from_stock GtkImage
    @brief Creates a GtkImage displaying a stock icon.
    @param stock_id a stock icon name (string)
    @param size (GtkIconSize)
    @return a new GtkImage displaying the stock icon

    Sample stock icon names are GTK_STOCK_OPEN, GTK_STOCK_QUIT. Sample stock sizes
    are GTK_ICON_SIZE_MENU, GTK_ICON_SIZE_SMALL_TOOLBAR. If the stock icon name isn't
    known, the image will be empty. You can register your own stock icon names,
    see gtk_icon_factory_add_default() and gtk_icon_factory_add().
 */
FALCON_FUNC Image::new_from_stock( VMARG )
{
    Item* i_id = vm->param( 0 );
    Item* i_sz = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || !i_id->isString()
        || !i_sz || !i_sz->isInteger() )
        throw_inv_params( "S,I" );
#endif
    AutoCString s( i_id->asString() );
    GtkWidget* img = gtk_image_new_from_stock(
            s.c_str(), (GtkIconSize) i_sz->asInteger() );
    vm->retval( new Image( vm->self().asClass(), (GtkImage*) img ) );
}


//FALCON_FUNC Image::new_from_animation( VMARG );

//FALCON_FUNC Image::new_from_icon_name( VMARG );

//FALCON_FUNC Image::new_from_gicon( VMARG );

/*#
    @method set_from_file GtkImage
    @brief Sets the image from a file.
    @param a filename or nil
 */
FALCON_FUNC Image::set_from_file( VMARG )
{
    Item* i_fnam = vm->param( 0 );
    // this method accepts nil
    gchar* fnam = NULL;
    MYSELF;
    GET_OBJ( self );
    if ( i_fnam )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !( i_fnam->isString() || i_fnam->isNil() ) )
            throw_inv_params( "S|nil" );
#endif
        if ( i_fnam->isString() )
        {
            // get the "raw" path
            String* filename = i_fnam->asString();

            Path path( *filename );
#ifdef FALCON_SYSTEM_WIN
            // if we are on windows, clear the path...
            filename->size( 0 );
            // and copy the winpath in it
            path.getWinFormat( *filename );
#else
            // otherwise, we copy the path returned via get()
            filename->copy( path.get() );
#endif

            AutoCString s( filename );
            gtk_image_set_from_file( (GtkImage*)_obj, s.c_str() );
            return;
        }
    }
    gtk_image_set_from_file( (GtkImage*)_obj, fnam );
}


/*#
    @method set_from_file GtkImage
    @brief Sets the image from a file.
    @param a filename or nil
 */
FALCON_FUNC Image::set_from_pixbuf( VMARG )
{
    Item* i_pixbuf = vm->param( 0 );

    MYSELF;
    GET_OBJ( self );

#ifndef NO_PARAMETER_CHECK
    if ( !i_pixbuf || !( i_pixbuf->isNil() || ( i_pixbuf->isObject() && IS_DERIVED( i_pixbuf, GdkPixbuf ) ) ) )
      throw_inv_params( "[GdkPixbuf]" );
#endif

    //    gtk_image_set_from_pixbuf( GET_PIXBUF( *i_pixbuf ) );
    gtk_image_set_from_pixbuf( (GtkImage*)_obj, GET_PIXBUF( *i_pixbuf ) );
}


//FALCON_FUNC Image::set_from_icon_set( VMARG );

//FALCON_FUNC Image::set_from_image( VMARG );

//FALCON_FUNC Image::set_from_pixbuf( VMARG );

//FALCON_FUNC Image::set_from_pixmap( VMARG );


/*#
    @method set_from_stock GtkImage
    @brief Sets an image from stock.
    @param stock_id a stock icon name (string)
    @param size (GtkIconSize)
 */
FALCON_FUNC Image::set_from_stock( VMARG )
{
    Item* i_id = vm->param( 0 );
    Item* i_sz = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || !i_id->isString()
        || !i_sz || !i_sz->isInteger() )
        throw_inv_params( "S,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString s( i_id->asString() );
    gtk_image_set_from_stock( (GtkImage*)_obj, s.c_str(),
        (GtkIconSize) i_sz->asInteger() );
}


//FALCON_FUNC Image::set_from_animation( VMARG );

//FALCON_FUNC Image::set_from_icon_name( VMARG );

//FALCON_FUNC Image::set_from_gicon( VMARG );

/*#
    @method clear GtkImage
    @brief Resets the image to be empty.
 */
FALCON_FUNC Image::clear( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_image_clear( (GtkImage*)_obj );
}


//FALCON_FUNC Image::set( VMARG );

//FALCON_FUNC Image::get( VMARG );

//FALCON_FUNC Image::set_pixel_size( VMARG );

//FALCON_FUNC Image::get_pixel_size( VMARG );


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
