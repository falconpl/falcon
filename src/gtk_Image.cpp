/**
 *  \file gtk_Image.cpp
 */

#include "gtk_Image.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Image::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Image = mod->addClass( "GtkImage", &Image::init );

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
    //{ "new_from_pixbuf",        &Image::foo },
    //{ "new_from_pixmap",        &Image::foo },
    //{ "new_from_stock",        &Image::foo },
    //{ "new_from_animation",        &Image::foo },
    //{ "new_from_icon_name",        &Image::foo },
    //{ "new_from_gicon",        &Image::foo },
    //{ "set_from_file",        &Image::foo },
    //{ "set_from_icon_set",        &Image::foo },
    //{ "set_from_image",        &Image::foo },
    //{ "set_from_pixbuf",        &Image::foo },
    //{ "set_from_pixmap",        &Image::foo },
    //{ "set_from_stock",        &Image::foo },
    //{ "set_from_animation",        &Image::foo },
    //{ "set_from_icon_name",        &Image::foo },
    //{ "set_from_gicon",        &Image::foo },
    //{ "clear",        &Image::foo },
    //{ "set",        &Image::foo },
    //{ "get",        &Image::foo },
    //{ "set_pixel_size",        &Image::foo },
    //{ "get_pixel_size",        &Image::foo },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Image, meth->name, meth->cb );
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
    Gtk::internal_add_slot( (GObject*) img );
    self->setUserData( new GData( (GObject*) img ) );
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

//FALCON_FUNC Image::new_from_pixbuf( VMARG );

//FALCON_FUNC Image::new_from_pixmap( VMARG );

//FALCON_FUNC Image::new_from_stock( VMARG );

//FALCON_FUNC Image::new_from_animation( VMARG );

//FALCON_FUNC Image::new_from_icon_name( VMARG );

//FALCON_FUNC Image::new_from_gicon( VMARG );

//FALCON_FUNC Image::set_from_file( VMARG );

//FALCON_FUNC Image::set_from_icon_set( VMARG );

//FALCON_FUNC Image::set_from_image( VMARG );

//FALCON_FUNC Image::set_from_pixbuf( VMARG );

//FALCON_FUNC Image::set_from_pixmap( VMARG );

//FALCON_FUNC Image::set_from_stock( VMARG );

//FALCON_FUNC Image::set_from_animation( VMARG );

//FALCON_FUNC Image::set_from_icon_name( VMARG );

//FALCON_FUNC Image::set_from_gicon( VMARG );

//FALCON_FUNC Image::clear( VMARG );

//FALCON_FUNC Image::set( VMARG );

//FALCON_FUNC Image::get( VMARG );

//FALCON_FUNC Image::set_pixel_size( VMARG );

//FALCON_FUNC Image::get_pixel_size( VMARG );


} // Gtk
} // Falcon
