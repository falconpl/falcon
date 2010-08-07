#ifndef GTK_TOGGLEBUTTON_HPP
#define GTK_TOGGLEBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ToggleButton
 */
class ToggleButton
    :
    public Gtk::CoreGObject
{
public:

    ToggleButton( const Falcon::CoreClass*, const GtkToggleButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_toggled( VMARG );

    static void on_toggled( GtkToggleButton*, gpointer );

    static FALCON_FUNC new_with_label( VMARG );

    static FALCON_FUNC new_with_mnemonic( VMARG );

    static FALCON_FUNC set_mode( VMARG );

    static FALCON_FUNC get_mode( VMARG );

    static FALCON_FUNC toggled( VMARG );

    static FALCON_FUNC get_active( VMARG );

    static FALCON_FUNC set_active( VMARG );

    static FALCON_FUNC get_inconsistent( VMARG );

    static FALCON_FUNC set_inconsistent( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TOGGLEBUTTON_HPP
