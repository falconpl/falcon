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
    Falcon::Symbol* c_Image = mod->addClass( "Image", &Image::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Misc" ) );
    c_Image->getClassDef()->addInheritance( in );
#if 0
    Gtk::MethodTab methods[] =
    {
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Image, meth->name, meth->cb );
#endif
}

/*#
    @class gtk.Image
    @brief A widget displaying an image

    The gtk.Image widget displays an image. Various kinds of object can be displayed
    as an image; most typically, you would load a GdkPixbuf ("pixel buffer") from a
    file, and then display that.

    [...]

    gtk.Image is a subclass of gtk.Misc, which implies that you can align it (center,
    left, right) and add padding to it, using gtk.Misc methods.

    gtk.Image is a "no window" widget (has no gdk.Window of its own), so by default
    does not receive events. If you want to receive events on the image, such as
    button clicks, place the image inside a gtk.EventBox, then connect to the event
    signals on the event box.
 */

/*#
    @init gtk.Image
    @brief Creates a new gtk.Image.
    @optparam filename a filename (string)
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


} // Gtk
} // Falcon
