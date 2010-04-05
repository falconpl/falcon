#ifndef GTK_SPINBUTTON_HPP
#define GTK_SPINBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::SpinButton
 */
class SpinButton
    :
    public Gtk::CoreGObject
{
public:

    SpinButton( const Falcon::CoreClass*, const GtkSpinButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_change_value( VMARG );

    static void on_change_value( GtkSpinButton*, GtkScrollType, gpointer );

    static FALCON_FUNC signal_input( VMARG );

    static gint on_input( GtkSpinButton*, gpointer, gpointer );

    static FALCON_FUNC signal_output( VMARG );

    static gboolean on_output( GtkSpinButton*, gpointer );

    static FALCON_FUNC signal_value_changed( VMARG );

    static void on_value_changed( GtkSpinButton*, gpointer );

    static FALCON_FUNC signal_wrapped( VMARG );

    static void on_wrapped( GtkSpinButton*, gpointer );

    //static FALCON_FUNC set_adjustment( VMARG );

    //static FALCON_FUNC get_adjustment( VMARG );

    static FALCON_FUNC set_digits( VMARG );

    static FALCON_FUNC set_increments( VMARG );

    static FALCON_FUNC set_range( VMARG );

    static FALCON_FUNC get_value_as_int( VMARG );

    static FALCON_FUNC set_value( VMARG );

    static FALCON_FUNC set_update_policy( VMARG );

    static FALCON_FUNC set_numeric( VMARG );

    static FALCON_FUNC spin( VMARG );

    static FALCON_FUNC set_wrap( VMARG );

    static FALCON_FUNC set_snap_to_ticks( VMARG );

    static FALCON_FUNC update( VMARG );

    static FALCON_FUNC get_digits( VMARG );

    static FALCON_FUNC get_increments( VMARG );

    static FALCON_FUNC get_numeric( VMARG );

    static FALCON_FUNC get_range( VMARG );

    static FALCON_FUNC get_snap_to_ticks( VMARG );

    static FALCON_FUNC get_update_policy( VMARG );

    static FALCON_FUNC get_value( VMARG );

    static FALCON_FUNC get_wrap( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_SPINBUTTON_HPP
