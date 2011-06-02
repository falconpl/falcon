#ifndef GTK_CHECKMENUITEM_HPP
#define GTK_CHECKMENUITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CheckMenuItem
 */
class CheckMenuItem
    :
    public Gtk::CoreGObject
{
public:

    CheckMenuItem( const Falcon::CoreClass*, const GtkCheckMenuItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_toggled( VMARG );

    static void on_toggled( GtkMenuItem*, gpointer );

    static FALCON_FUNC new_with_label( VMARG );

    static FALCON_FUNC new_with_mnemonic( VMARG );

    //static FALCON_FUNC set_state( VMARG );

    static FALCON_FUNC get_active( VMARG );

    static FALCON_FUNC set_active( VMARG );

    //static FALCON_FUNC set_show_toggle( VMARG );

    static FALCON_FUNC toggled( VMARG );

    static FALCON_FUNC get_inconsistent( VMARG );

    static FALCON_FUNC set_inconsistent( VMARG );

    static FALCON_FUNC set_draw_as_radio( VMARG );

    static FALCON_FUNC get_draw_as_radio( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CHECKMENUITEM_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
