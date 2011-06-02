#ifndef GTK_SCALEBUTTON_HPP
#define GTK_SCALEBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ScaleButton
 */
class ScaleButton
    :
    public Gtk::CoreGObject
{
public:

    ScaleButton( const Falcon::CoreClass*, const GtkScaleButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_popdown( VMARG );

    static void on_popdown( GtkScaleButton*, gpointer );

    static FALCON_FUNC signal_popup( VMARG );

    static void on_popup( GtkScaleButton*, gpointer );

    static FALCON_FUNC signal_value_changed( VMARG );

    static void on_value_changed( GtkScaleButton*, gdouble, gpointer );

    static FALCON_FUNC set_adjustment( VMARG );

    static FALCON_FUNC set_icons( VMARG );

    static FALCON_FUNC set_value( VMARG );

    static FALCON_FUNC get_adjustment( VMARG );

    static FALCON_FUNC get_value( VMARG );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    static FALCON_FUNC get_popup( VMARG );

    static FALCON_FUNC get_plus_button( VMARG );

    static FALCON_FUNC get_minus_button( VMARG );

#if 0 // deprecated
    static FALCON_FUNC set_orientation( VMARG );
    static FALCON_FUNC get_orientation( VMARG );
#endif
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_SCALEBUTTON_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
