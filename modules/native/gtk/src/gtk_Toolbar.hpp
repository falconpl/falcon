#ifndef GTK_TOOLBAR_HPP
#define GTK_TOOLBAR_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Toolbar
 */
class Toolbar
    :
    public Gtk::CoreGObject
{
public:

    Toolbar( const Falcon::CoreClass*, const GtkToolbar* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

#if 0 // internals
    static FALCON_FUNC signal_focus_home_or_end( VMARG );

    static void on_focus_home_or_end( GtkToolbar*, gpointer );
#endif

    static FALCON_FUNC signal_orientation_changed( VMARG );

    static void on_orientation_changed( GtkToolbar*, GtkOrientation, gpointer );

    static FALCON_FUNC signal_popup_context_menu( VMARG );

    static void on_popup_context_menu( GtkToolbar*, gint, gint, gint, gpointer );

    static FALCON_FUNC signal_style_changed( VMARG );

    static void on_style_changed( GtkToolbar*, GtkToolbarStyle, gpointer );

    static FALCON_FUNC insert( VMARG );

    static FALCON_FUNC get_item_index( VMARG );

    static FALCON_FUNC get_n_items( VMARG );

    //static FALCON_FUNC get_nth_item( VMARG ); TODO

    static FALCON_FUNC get_drop_index( VMARG );

    static FALCON_FUNC set_drop_highlight_item( VMARG );

    static FALCON_FUNC set_show_arrow( VMARG );

#if 0 // deprecated
    static FALCON_FUNC set_orientation( VMARG );

    static FALCON_FUNC set_tooltips( VMARG );
#endif

    static FALCON_FUNC unset_icon_size( VMARG );

    static FALCON_FUNC get_show_arrow( VMARG );

    //static FALCON_FUNC get_orientation( VMARG );

    static FALCON_FUNC get_style( VMARG );

    static FALCON_FUNC get_icon_size( VMARG );

    //static FALCON_FUNC get_tooltips( VMARG );

    static FALCON_FUNC get_relief_style( VMARG );

#if 0 // deprecated
    static FALCON_FUNC append_item( VMARG );

    static FALCON_FUNC prepend_item( VMARG );

    static FALCON_FUNC insert_item( VMARG );

    static FALCON_FUNC append_space( VMARG );

    static FALCON_FUNC prepend_space( VMARG );

    static FALCON_FUNC insert_space( VMARG );

    static FALCON_FUNC append_element( VMARG );

    static FALCON_FUNC prepend_element( VMARG );

    static FALCON_FUNC insert_element( VMARG );

    static FALCON_FUNC append_widget( VMARG );

    static FALCON_FUNC prepend_widget( VMARG );

    static FALCON_FUNC insert_widget( VMARG );
#endif

    static FALCON_FUNC set_style( VMARG );

    //static FALCON_FUNC insert_stock( VMARG );

    static FALCON_FUNC set_icon_size( VMARG );

    //static FALCON_FUNC remove_space( VMARG );

    static FALCON_FUNC unset_style( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TOOLBAR_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
