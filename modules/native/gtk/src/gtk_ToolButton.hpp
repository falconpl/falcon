#ifndef GTK_TOOLBUTTON_HPP
#define GTK_TOOLBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ToolButton
 */
class ToolButton
    :
    public Gtk::CoreGObject
{
public:

    ToolButton( const Falcon::CoreClass*, const GtkToolButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_clicked( VMARG );

    static void on_clicked( GtkToolButton*, gpointer );

    static FALCON_FUNC new_from_stock( VMARG );

    static FALCON_FUNC set_label( VMARG );

    static FALCON_FUNC get_label( VMARG );

    static FALCON_FUNC set_use_underline( VMARG );

    static FALCON_FUNC get_use_underline( VMARG );

    static FALCON_FUNC set_stock_id( VMARG );

    static FALCON_FUNC get_stock_id( VMARG );

    static FALCON_FUNC set_icon_name( VMARG );

    static FALCON_FUNC get_icon_name( VMARG );

    static FALCON_FUNC set_icon_widget( VMARG );

    static FALCON_FUNC get_icon_widget( VMARG );

    static FALCON_FUNC set_label_widget( VMARG );

    static FALCON_FUNC get_label_widget( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TOOLBUTTON_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
