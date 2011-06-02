#ifndef GTK_COLORBUTTON_HPP
#define GTK_COLORBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ColorButton
 */
class ColorButton
    :
    public Gtk::CoreGObject
{
public:

    ColorButton( const Falcon::CoreClass*, const GtkColorButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_color_set( VMARG );

    static void on_color_set( GtkColorButton*, gpointer );

    static FALCON_FUNC new_with_color( VMARG );

    static FALCON_FUNC set_color( VMARG );

    static FALCON_FUNC get_color( VMARG );

    static FALCON_FUNC set_alpha( VMARG );

    static FALCON_FUNC get_alpha( VMARG );

    static FALCON_FUNC set_use_alpha( VMARG );

    static FALCON_FUNC get_use_alpha( VMARG );

    static FALCON_FUNC set_title( VMARG );

    static FALCON_FUNC get_title( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_COLORBUTTON_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
