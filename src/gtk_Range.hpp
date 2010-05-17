#ifndef GTK_RANGE_HPP
#define GTK_RANGE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Range
 */
class Range
    :
    public Gtk::CoreGObject
{
public:

    Range( const Falcon::CoreClass*, const GtkRange* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC signal_adjust_bounds( VMARG );

    static void on_adjust_bounds( GtkRange*, gdouble, gpointer );

    static FALCON_FUNC signal_change_value( VMARG );

    static gboolean on_change_value( GtkRange*, GtkScrollType, gdouble, gpointer );

    static FALCON_FUNC signal_move_slider( VMARG );

    static void on_move_slider( GtkRange*, GtkScrollType, gpointer );

    static FALCON_FUNC signal_value_changed( VMARG );

    static void on_value_changed( GtkRange*, gpointer );

    static FALCON_FUNC get_fill_level( VMARG );

    static FALCON_FUNC get_restrict_to_fill_level( VMARG );

    static FALCON_FUNC get_show_fill_level( VMARG );

    static FALCON_FUNC set_fill_level( VMARG );

    static FALCON_FUNC set_restrict_to_fill_level( VMARG );

    static FALCON_FUNC set_show_fill_level( VMARG );

    static FALCON_FUNC get_adjustment( VMARG );

    static FALCON_FUNC set_update_policy( VMARG );

    static FALCON_FUNC set_adjustment( VMARG );

    static FALCON_FUNC get_inverted( VMARG );

    static FALCON_FUNC set_inverted( VMARG );

    static FALCON_FUNC get_update_policy( VMARG );

    static FALCON_FUNC get_value( VMARG );

    static FALCON_FUNC set_increments( VMARG );

    static FALCON_FUNC set_range( VMARG );

    static FALCON_FUNC set_value( VMARG );

    static FALCON_FUNC set_lower_stepper_sensitivity( VMARG );

    static FALCON_FUNC get_lower_stepper_sensitivity( VMARG );

    static FALCON_FUNC set_upper_stepper_sensitivity( VMARG );

    static FALCON_FUNC get_upper_stepper_sensitivity( VMARG );

#if GTK_MINOR_VERSION >= 18
    static FALCON_FUNC get_flippable( VMARG );

    static FALCON_FUNC set_flippable( VMARG );
#endif

#if GTK_MINOR_VERSION >= 20
    static FALCON_FUNC get_min_slider_size( VMARG );

    //static FALCON_FUNC get_range_rect( VMARG );

    static FALCON_FUNC get_slider_range( VMARG );

    static FALCON_FUNC get_slider_size_fixed( VMARG );

    //static FALCON_FUNC set_min_slider_size( VMARG );

    static FALCON_FUNC set_slider_size_fixed( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_RANGE_HPP
