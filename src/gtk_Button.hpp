#ifndef GTK_BUTTON_HPP
#define GTK_BUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Button
 */
class Button
    :
    public Gtk::CoreGObject
{
public:

    Button( const Falcon::CoreClass*, const GtkButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_activate( VMARG );

    static void on_activate( GtkButton*, gpointer );

    static FALCON_FUNC signal_clicked( VMARG );

    static void on_clicked( GtkButton*, gpointer );

    static FALCON_FUNC signal_enter( VMARG );

    static void on_enter( GtkButton*, gpointer );

    static FALCON_FUNC signal_leave( VMARG );

    static void on_leave( GtkButton*, gpointer );

    static FALCON_FUNC signal_pressed( VMARG );

    static void on_pressed( GtkButton*, gpointer );

    static FALCON_FUNC signal_released( VMARG );

    static void on_released( GtkButton*, gpointer );

    static FALCON_FUNC new_with_label( VMARG );

    static FALCON_FUNC new_with_mnemonic( VMARG );

    static FALCON_FUNC new_from_stock( VMARG );

    static FALCON_FUNC pressed( VMARG );

    static FALCON_FUNC released( VMARG );

    static FALCON_FUNC clicked( VMARG );

    static FALCON_FUNC enter( VMARG );

    static FALCON_FUNC leave( VMARG );

    static FALCON_FUNC set_relief( VMARG );

    static FALCON_FUNC get_relief( VMARG );

    static FALCON_FUNC set_label( VMARG );

    static FALCON_FUNC get_label( VMARG );

    static FALCON_FUNC set_use_stock( VMARG );

    static FALCON_FUNC get_use_stock( VMARG );

    static FALCON_FUNC set_use_underline( VMARG );

    static FALCON_FUNC get_use_underline( VMARG );

    static FALCON_FUNC set_focus_on_click( VMARG );

    static FALCON_FUNC get_focus_on_click( VMARG );

    static FALCON_FUNC set_alignment( VMARG );

    static FALCON_FUNC get_alignment( VMARG );

    static FALCON_FUNC set_image( VMARG );

    static FALCON_FUNC get_image( VMARG );

    static FALCON_FUNC set_image_position( VMARG );

    static FALCON_FUNC get_image_position( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_BUTTON_HPP
