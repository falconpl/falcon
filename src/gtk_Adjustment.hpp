#ifndef GTK_ADJUSTMENT_HPP
#define GTK_ADJUSTMENT_HPP

#include "modgtk.hpp"

#define GET_ADJUSTMENT( item ) \
        ((GtkAdjustment*) Falcon::dyncast<Gtk::Adjustment*>( (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Adjustment
 */
class Adjustment
    :
    public Gtk::CoreGObject
{
public:

    static void modInit( Falcon::Module* );

    Adjustment( const Falcon::CoreClass*, const GtkAdjustment* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_changed( VMARG );

    static void on_changed( GtkAdjustment*, gpointer );

    static FALCON_FUNC signal_value_changed( VMARG );

    static void on_value_changed( GtkAdjustment*, gpointer );

    static FALCON_FUNC get_value( VMARG );

    static FALCON_FUNC set_value( VMARG );

    static FALCON_FUNC clamp_page( VMARG );

    static FALCON_FUNC changed( VMARG );

    static FALCON_FUNC value_changed( VMARG );

#if GTK_MINOR_VERSION >= 14
    static FALCON_FUNC configure( VMARG );

    static FALCON_FUNC get_lower( VMARG );

    static FALCON_FUNC get_page_increment( VMARG );

    static FALCON_FUNC get_page_size( VMARG );

    static FALCON_FUNC get_step_increment( VMARG );

    static FALCON_FUNC get_upper( VMARG );

    static FALCON_FUNC set_lower( VMARG );

    static FALCON_FUNC set_page_increment( VMARG );

    static FALCON_FUNC set_page_size( VMARG );

    static FALCON_FUNC set_step_increment( VMARG );

    static FALCON_FUNC set_upper( VMARG );
#endif // GTK_MINOR_VERSION >= 14

};


} // Gtk
} // Falcon

#endif // !GTK_ADJUSTMENT_HPP
