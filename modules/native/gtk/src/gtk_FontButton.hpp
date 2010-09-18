#ifndef GTK_FONTBUTTON_HPP
#define GTK_FONTBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::FontButton
 */
class FontButton
    :
    public Gtk::CoreGObject
{
public:

    FontButton( const Falcon::CoreClass*, const GtkFontButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_font_set( VMARG );

    static void on_font_set( GtkFontButton*, gpointer );

    static FALCON_FUNC new_with_font( VMARG );

    static FALCON_FUNC set_font_name( VMARG );

    static FALCON_FUNC get_font_name( VMARG );

    static FALCON_FUNC set_show_style( VMARG );

    static FALCON_FUNC get_show_style( VMARG );

    static FALCON_FUNC set_show_size( VMARG );

    static FALCON_FUNC get_show_size( VMARG );

    static FALCON_FUNC set_use_font( VMARG );

    static FALCON_FUNC get_use_font( VMARG );

    static FALCON_FUNC set_use_size( VMARG );

    static FALCON_FUNC get_use_size( VMARG );

    static FALCON_FUNC set_title( VMARG );

    static FALCON_FUNC get_title( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_FONTBUTTON_HPP
