/**
 *  \file gtk_Widget.cpp
 */

#include "gtk_Widget.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {


void Widget::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Widget = mod->addClass( "Widget", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Object" ) );
    c_Widget->getClassDef()->addInheritance( in );

    c_Widget->setWKS( true );
    c_Widget->getClassDef()->factory( &Widget::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_accel_closures_changed",  &Widget::signal_accel_closures_changed },
    //{ "button_press_event",             &Widget::signal_button_press_event },
    //{ "button_release_event",           &Widget::signal_button_release_event },
    { "signal_can_activate_accel",      &Widget::signal_can_activate_accel },
    //{ "signal_child_notify",            &Widget::signal_child_notify },
    //{ "signal_client_event",            &Widget::signal_client_event },
    { "signal_composited_changed",      &Widget::signal_composited_changed },
    //{ "signal_configure_event",         &Widget::signal_configure_event },
    //{ "signal_damage_event",            &Widget::signal_damage_event },
    { "signal_delete_event",            &Widget::signal_delete_event },
    { "signal_destroy_event",           &Widget::signal_destroy_event },
    //{ "signal_direction_changed",       &Widget::signal_direction_changed },
    //{ "signal_drag_begin",              &Widget::signal_drag_begin },
    //{ "signal_drag_data_delete",        &Widget::signal_drag_data_delete },
    //{ "signal_drag_data_get",           &Widget::signal_drag_data_get },
    //{ "signal_drag_data_received",      &Widget::signal_drag_data_received },
    //{ "signal_drag_drop",               &Widget::signal_drag_drop },
    //{ "signal_drag_end",                &Widget::signal_drag_end },
    //{ "signal_drag_failed",             &Widget::signal_drag_failed },
    //{ "signal_drag_leave",              &Widget::signal_drag_leave },
    //{ "signal_drag_motion",             &Widget::signal_drag_motion },
    //{ "signal_enter_notify_event",      &Widget::signal_enter_notify_event },
    //{ "signal_event",                   &Widget::signal_event },
    //{ "signal_event_after",             &Widget::signal_event_after },
    //{ "signal_expose_event",            &widget::signal_expose_event },
    //{ "signal_focus",                   &Widget::signal_focus },
    //{ "signal_focus_in_event",          &Widget::signal_focus_in_event },
    //{ "signal_focus_out_event",         &Widget::signal_focus_out_event },
    //{ "signal_grab_broken_event",       &Widget::signal_grab_broken_event },
    //{ "signal_grab_focus",              &Widget::signal_grab_focus },
    //{ "signal_grab_notify",             &Widget::signal_grab_notify },
    { "signal_hide",                    &Widget::signal_hide },
    //{ "signal_hierarchy_changed",       &Widget::signal_hierarchy_changed },
    //{ "signal_key_press_event",         &Widget::signal_key_press_event },
    //{ "signal_key_release_event",       &Widget::signal_key_release_event },
    //{ "signal_keynav_failed",           &Widget::signal_keynav_failed },
    //{ "signal_leave_notify_event",      &Widget::signal_leave_notify_event },
    //{ "signal_map",                     &Widget::signal_map },
    //{ "signal_map_event",               &Widget::signal_map_event },
    //{ "signal_mnemonic_activate",       &Widget::signal_mnemonic_activate },
    //{ "signal_motion_notify_event",     &Widget::signal_motion_notify_event },
    //{ "signal_move_focus",              &Widget::signal_move_focus },
    //{ "signal_no_expose_event",         &Widget::signal_no_expose_event },
    //{ "signal_parent_set",              &Widget::signal_parent_set },
    //{ "signal_popup_menu",              &Widget::signal_popup_menu },
    //{ "signal_property_notify_event",   &Widget::signal_property_notify_event },
    //{ "signal_proximity_in_event",      &Widget::signal_proximity_in_event },
    //{ "signal_proximity_out_event",     &Widget::signal_proximity_out_event },
    //{ "signal_query_tooltip",           &Widget::signal_query_tooltip },
    //{ "signal_realize",                 &Widget::signal_realize },
    //{ "signal_screen_changed",          &Widget::signal_screen_changed },
    //{ "signal_scroll_event",            &Widget::signal_scroll_event },
    //{ "signal_selection_clear_event",   &Widget::signal_selection_clear_event },
    //{ "signal_selection_get",           &Widget::signal_selection_get },
    //{ "signal_selection_notify_event",  &Widget::signal_selection_notify_event },
    //{ "signal_selection_received",      &Widget::signal_selection_received },
    //{ "signal_selection_request_event", &Widget::signal_selection_request_event },
    { "signal_show",                    &Widget::signal_show },
    //{ "signal_show_help",               &Widget::signal_show_help },
    //{ "signal_size_allocate",           &Widget::signal_size_allocate },
    //{ "signal_size_request",            &Widget::signal_size_request },
    //{ "signal_state_changed",           &Widget::signal_state_changed },
    //{ "signal_style_set",               &Widget::signal_style_set },
    //{ "signal_unmap",                   &Widget::signal_unmap },
    //{ "signal_unmap_event",             &Widget::signal_unmap_event },
    //{ "signal_unrealize",               &Widget::signal_unrealize },
    //{ "signal_visibility_notify_event", &Widget::signal_visibility_notify_event },
    //{ "signal_window_state_event",      &Widget::signal_window_state_event },

    { "show",                   &Widget::show },
    { "show_now",               &Widget::show_now },
    { "hide",                   &Widget::hide },
    { "show_all",               &Widget::show_all },
    { "hide_all",               &Widget::hide_all },
    { "reparent",               &Widget::reparent },
    { "is_focus",               &Widget::is_focus },
    { "grab_focus",             &Widget::grab_focus },
    { "grab_default",           &Widget::grab_default },
    { "set_name",               &Widget::set_name },
    { "get_name",               &Widget::get_name },
    { "set_sensitive",          &Widget::set_sensitive },
    { "get_toplevel",           &Widget::set_sensitive },
    { "get_events",             &Widget::get_events },
    { "is_ancestor",            &Widget::is_ancestor },
    { "hide_on_delete",         &Widget::hide_on_delete },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Widget, meth->name, meth->cb );
}


Widget::Widget( const Falcon::CoreClass* gen, const GtkWidget* wdt )
    :
    Falcon::CoreObject( gen )
{
    if ( wdt )
        setUserData( new GData( (GObject*) wdt ) );
}

bool Widget::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    return defaultProperty( s, it );
}


bool Widget::setProperty( const Falcon::String&, const Falcon::Item& )
{
    return false;
}


Falcon::CoreObject* Widget::factory( const Falcon::CoreClass* gen, void* wdt, bool )
{
    return new Widget( gen, (GtkWidget*) wdt );
}


/*#
    @class gtk.Widget
    @brief Base class for all widgets
    @raise GtkError on direct instanciation

    GtkWidget is the base class all widgets in GTK+ derive from.
    It manages the widget lifecycle, states and style.
 */

/*#
    @init gtk.Widget
    @brief Base class for all widgets
    @raise GtkError on direct instanciation
 */


/*#
    @method signal_accel_closures_changed gtk.Widget
    @brief Connect a VMSlot to the widget accel_closures_changed signal and return it
 */
FALCON_FUNC Widget::signal_accel_closures_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "accel_closures_changed",
        (void*) &Widget::on_accel_closures_changed, vm );
}


void Widget::on_accel_closures_changed( GtkWidget* wdt, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) wdt, "accel_closures_changed",
        "on_accel_closures_changed", (VMachine*)_vm );
}


//FALCON_FUNC signal_button_press_event( VMARG );

//gboolean on_button_press_event( GtkWidget*, GdkEventButton*, gpointer );

//FALCON_FUNC signal_button_release_event( VMARG );

//gboolean on_button_release_event( GtkWidget*, GdkEventButton*, gpointer );



/*#
    @method signal_can_activate_accel gtk.Widget
    @brief Connect a VMSlot to the widget can_activate_accel signal and return it

    Determines whether an accelerator that activates the signal identified by
    signal_id can currently be activated. This signal is present to allow applications
    and derived widgets to override the default GtkWidget handling for determining
    whether an accelerator can be activated.

    The callback function must return a boolean (returning true if the signal can
    be activated). It will get an integer as parameter, that is the ID of a signal
    installed on widget.
 */
FALCON_FUNC Widget::signal_can_activate_accel( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "can_activate_accel",
        (void*) &Widget::on_can_activate_accel, vm );
}


gboolean Widget::on_can_activate_accel( GtkWidget* obj, guint signal_id, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "can_activate_accel", false );

    if ( !cs || cs->empty() )
        return TRUE; // can activate

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_can_activate_accel", it ) )
            {
                printf(
                "[Widget::on_can_activate_accel] invalid callback (expected callable)\n" );
                return FALSE; // block event
            }
        }
        vm->pushParam( (int64) signal_id );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( !it.asBoolean() )
                return FALSE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[Widget::on_can_activate_accel] invalid callback (expected boolean)\n" );
            return FALSE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return TRUE; // can activate
}


//FALCON_FUNC Widget::signal_child_notify( VMARG );

//gboolean Widget::on_child_notify( GtkWidget*, GParamSpec*, gpointer );

//FALCON_FUNC Widget::signal_client_event( VMARG );

//gboolean Widget::on_client_event( GtkWidget*, GdkEventClient*, gpointer );


/*#
    @method signal_composited_changed gtk.Widget
    @brief Connect a VMSlot to the widget composited_changed signal and return it

    The composited-changed signal is emitted when the composited status of
    widgets screen changes.
 */
FALCON_FUNC Widget::signal_composited_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "composited_changed",
        (void*) &Widget::on_composited_changed, vm );
}


void Widget::on_composited_changed( GtkWidget* obj, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) obj, "composited_changed",
        "on_composited_changed", (VMachine*)_vm );
}


//FALCON_FUNC Widget::signal_configure_event( VMARG );

//gboolean Widget::on_configure_event( GtkWidget*, GdkEventConfigure*, gpointer );

//FALCON_FUNC signal_damage_event( VMARG );

//gboolean on_damage_event( GtkWidget*, GdkEvent*, gpointer );


/*#
    @method signal_delete_event gtk.Widget
    @brief Connect a VMSlot to the widget delete_event signal and return it

    The callback function must return a boolean, that if is true, will block the event.

    The delete-event signal is emitted if a user requests that a toplevel window
    is closed. The default handler for this signal destroys the window.
    Connecting gtk_widget_hide_on_delete() to this signal will cause the window
    to be hidden instead, so that it can later be shown again without reconstructing it.
 */
FALCON_FUNC Widget::signal_delete_event( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "delete_event", (void*) &Widget::on_delete_event, vm );
}


gboolean Widget::on_delete_event( GtkWidget* obj, GdkEvent* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "delete_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_delete_event", it ) )
            {
                printf(
                "[Widget::on_delete_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        //vm->pushParam( Item( (int64)((GdkEventAny*)ev)->type ) );
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[Widget::on_delete_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


/*#
    @method signal_destroy_event gtk.Widget
    @brief Connect a VMSlot to the widget destroy signal and return it

    The callback function must return a boolean, true to stop other handlers from
    being invoked for the event, false to propagate the event further.

    The destroy-event signal is emitted when a GdkWindow is destroyed.
    You rarely get this signal, because most widgets disconnect themselves from
    their window before they destroy it, so no widget owns the window at destroy time.

    To receive this signal, the GdkWindow associated to the widget needs to enable
    the GDK_STRUCTURE_MASK mask. GDK will enable this mask automatically for all new windows.
 */
FALCON_FUNC Widget::signal_destroy_event( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "destroy_event", (void*) &Widget::on_destroy_event, vm );
}


gboolean Widget::on_destroy_event( GtkWidget* obj, GdkEvent*, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "destroy_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_destroy_event", it ) )
            {
                printf(
                "[Widget::on_destroy_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        //vm->pushParam( Item( (int64)((GdkEventAny*)ev)->type ) );
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[Widget::on_destroy_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


//FALCON_FUNC Widget::signal_direction_changed( VMARG );

//void Widget::on_direction_changed( GtkWidget*, GtkTextDirection, gpointer );

//FALCON_FUNC Widget::signal_drag_begin( VMARG );

//void Widget::on_drag_begin( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_data_delete( VMARG );

//void Widget::on_drag_data_delete( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_data_get( VMARG );

//void Widget::on_drag_data_get( GtkWidget*, GdkDragContext*, GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_data_received( VMARG );

//void Widget::on_drag_data_received( GtkWidget*, GdkDragContext*, gint, gint,
        //GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_drop( VMARG );

//gboolean Widget::on_drag_drop( GtkWidget*, GdkDragContext*, gint, gint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_end( VMARG );

//void Widget::on_drag_end( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_failed( VMARG );

//gboolean Widget::on_drag_failed( GtkWidget*, GdkDragContext*, GtkDragResult, gpointer );

//FALCON_FUNC Widget::signal_drag_leave( VMARG );

//void Widget::on_drag_leave( GtkWidget*, GdkDragContext*, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_motion( VMARG );

//gboolean Widget::on_drag_motion( GtkWidget*, GdkDragContext*, gint, gint, guint, gpointer );

//FALCON_FUNC Widget::signal_enter_notify_event( VMARG );

//gboolean Widget::on_enter_notify_event( GtkWidget*, GdkEventCrossing*, gpointer );

//FALCON_FUNC Widget::signal_event( VMARG );

//gboolean Widget::on_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_event_after( VMARG );

//void Widget::on_event_after( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_expose_event( VMARG );

//gboolean Widget::on_expose_event( GtkWidget*, GdkEventExpose*, gpointer );

//FALCON_FUNC Widget::signal_focus( VMARG );

//gboolean Widget::on_focus( GtkWidget*, GtkDirectionType*, gpointer );

//FALCON_FUNC Widget::signal_focus_in_event( VMARG );

//gboolean Widget::on_focus_in_event( GtkWidget*, GdkEventFocus*, gpointer );

//FALCON_FUNC Widget::signal_focus_out_event( VMARG );

//gboolean Widget::on_focus_out_event( GtkWidget*, GdkEventFocus*, gpointer );

//FALCON_FUNC Widget::signal_grab_broken_event( VMARG );

//gboolean Widget::on_grab_broken_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_grab_focus( VMARG );

//void Widget::on_grab_focus( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_grab_notify( VMARG );

//void Widget::on_grab_notify( GtkWidget*, gboolean, gpointer );


/*#
    @method signal_hide gtk.Widget
    @brief Connect a VMSlot to the widget hide signal and return it
 */
FALCON_FUNC Widget::signal_hide( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "hide", (void*) &Widget::on_hide, vm );
}


void Widget::on_hide( GtkWidget* obj, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) obj, "hide", "on_hide", (VMachine*)_vm );
}




//FALCON_FUNC Widget::signal_hierarchy_changed( VMARG );

//void Widget::on_hierarchy_changed( GtkWidget*, GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_key_press_event( VMARG );

//gboolean Widget::on_key_press_event( GtkWidget*, GdkEventKey*, gpointer );

//FALCON_FUNC Widget::signal_key_release_event( VMARG );

//gboolean Widget::on_key_release_event( GtkWidget*, GdkEventKey*, gpointer );

//FALCON_FUNC Widget::signal_keynav_failed( VMARG );

//gboolean Widget::on_keynav_failed( GtkWidget*, GtkDirectionType, gpointer );

//FALCON_FUNC Widget::signal_leave_notify_event( VMARG );

//gboolean Widget::on_leave_notify_event( GtkWidget*, GdkEventCrossing*, gpointer );

//FALCON_FUNC Widget::signal_map( VMARG );

//void Widget::on_map( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_map_event( VMARG );

//gboolean Widget::on_map_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_mnemonic_activate( VMARG );

//gboolean Widget::on_mnemonic_activate( GtkWidget*, gboolean, gpointer );

//FALCON_FUNC Widget::signal_motion_notify_event( VMARG );

//gboolean Widget::on_motion_notify_event( GtkWidget*, GdkEventMotion*, gpointer );

//FALCON_FUNC Widget::signal_move_focus( VMARG );

//void Widget::on_move_focus( GtkWidget*, GtkDirectionType, gpointer );

//FALCON_FUNC Widget::signal_no_expose_event( VMARG );

//gboolean Widget::on_no_expose_event( GtkWidget*, GdkEventNoExpose*, gpointer );

//FALCON_FUNC Widget::signal_parent_set( VMARG );

//void Widget::on_parent_set( GtkWidget*, GtkObject*, gpointer );

//FALCON_FUNC Widget::signal_popup_menu( VMARG );

//gboolean Widget::on_popup_menu( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_property_notify_event( VMARG );

//gboolean Widget::on_property_notify_event( GtkWidget*, GdkEventProperty*, gpointer );

//FALCON_FUNC Widget::signal_proximity_in_event( VMARG );

//gboolean Widget::on_proximity_in_event( GtkWidget*, GdkEventProximity*, gpointer );

//FALCON_FUNC Widget::signal_proximity_out_event( VMARG );

//gboolean Widget::on_proximity_out_event( GtkWidget*, GdkEventProximity*, gpointer );

//FALCON_FUNC Widget::signal_query_tooltip( VMARG );

//gboolean Widget::on_query_tooltip( GtkWidget*, gint, gint, gboolean, GtkTooltip*, gpointer );

//FALCON_FUNC Widget::signal_realize( VMARG );

//void Widget::on_realize( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_screen_changed( VMARG );

//void Widget::on_screen_changed( GtkWidget*, GdkScreen*, gpointer );

//FALCON_FUNC Widget::signal_scroll_event( VMARG );

//gboolean Widget::on_scroll_event( GtkWidget*, GdkEventScroll*, gpointer );

//FALCON_FUNC Widget::signal_selection_clear_event( VMARG );

//gboolean Widget::on_selection_clear_event( GtkWidget*, GdkEventSelection*, gpointer );

//FALCON_FUNC Widget::signal_selection_get( VMARG );

//void Widget::on_selection_get( GtkWidget*, GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_selection_notify_event( VMARG );

//gboolean Widget::on_selection_notify_event( GtkWidget*, GdkEventSelection*, gpointer );

//FALCON_FUNC Widget::signal_selection_received( VMARG );

//void Widget::on_selection_received( GtkWidget*, GtkSelectionData*, guint, gpointer );

//FALCON_FUNC Widget::signal_selection_request_event( VMARG );

//gboolean Widget::on_selection_request_event( GtkWidget*, GdkEventSelection*, gpointer );


/*#
    @method signal_show gtk.Widget
    @brief Connect a VMSlot to the widget show signal and return it
 */
FALCON_FUNC Widget::signal_show( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "show", (void*) &Widget::on_show, vm );
}


void Widget::on_show( GtkWidget* obj, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) obj, "show", "on_show", (VMachine*)_vm );
}


//FALCON_FUNC Widget::signal_show_help( VMARG );

//gboolean Widget::on_show_help( GtkWidget*, GtkWidgetHelpType, gpointer );

//FALCON_FUNC Widget::signal_size_allocate( VMARG );

//void Widget::on_size_allocate( GtkWidget*, GtkAllocation*, gpointer );

//FALCON_FUNC Widget::signal_size_request( VMARG );

//void Widget::on_size_request( GtkWidget*, GtkRequisition*, gpointer );

//FALCON_FUNC Widget::signal_state_changed( VMARG );

//void Widget::on_state_changed( GtkWidget*, GtkState, gpointer );

//FALCON_FUNC Widget::signal_style_set( VMARG );

//void Widget::on_style_set( GtkWidget*, GtkStyle*, gpointer );

//FALCON_FUNC Widget::signal_unmap( VMARG );

//void Widget::on_unmap( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_unmap_event( VMARG );

//gboolean Widget::on_unmap_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_unrealize( VMARG );

//void Widget::on_unrealize( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_visibility_notify_event( VMARG );

//gboolean Widget::on_visibility_notify_event( GtkWidget*, GdkEventVisibility*, gpointer );

//FALCON_FUNC Widget::signal_window_state_event( VMARG );

//gboolean Widget::on_window_state_event( GtkWidget*, GdkEventWindowState*, gpointer );


/*#
    @method show gtk.Widget
    @brief Flags a widget to be displayed.
 */
FALCON_FUNC Widget::show( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show( ((GtkWidget*)_obj) );
}


/*#
    @method show_now gtk.Widget
    @brief Shows a widget.
 */
FALCON_FUNC Widget::show_now( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show_now( ((GtkWidget*)_obj) );
}


/*#
    @method hide gtk.Widget
    @brief Reverses the effects of show(), causing the widget to be hidden (invisible to the user).
 */
FALCON_FUNC Widget::hide( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_hide( ((GtkWidget*)_obj) );
}


/*#
    @method show_all gtk.Widget
    @brief Recursively shows a widget, and any child widgets (if the widget is a container).
 */
FALCON_FUNC Widget::show_all( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show_all( ((GtkWidget*)_obj) );
}


/*#
    @method hide_all gtk.Widget
    @brief Recursively hides a widget and any child widgets.
 */
FALCON_FUNC Widget::hide_all( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_hide_all( ((GtkWidget*)_obj) );
}


/*#
    @method activate gtk.Widget
    @brief For widgets that can be "activated" (buttons, menu items, etc).
    @return boolean
 */
FALCON_FUNC Widget::activate( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_widget_activate( ((GtkWidget*)_obj) ) );
}


/*#
    @method reparent gtk.Widget
    @brief Moves a widget from one container to another.
    @param new_parent The new parent
 */
FALCON_FUNC Widget::reparent( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() ||
        !( i_wdt->isOfClass( "Widget" ) || i_wdt->isOfClass( "gtk.Widget" ) ) )
    {
        throw_inv_params( "Widget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    gtk_widget_reparent( (GtkWidget*)_obj, (GtkWidget*) wdt );
}


/*#
    @method is_focus gtk.Widget
    @brief Determines if the widget is the focus widget within its toplevel.
    This does not mean that the HAS_FOCUS flag is necessarily set;
    HAS_FOCUS will only be set if the toplevel widget additionally has the
    global input focus.
    @return (boolean)
 */
FALCON_FUNC Widget::is_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_widget_is_focus( (GtkWidget*)_obj ) );
}


/*#
    @method grab_focus gtk.Widget
    @brief Causes widget to have the keyboard focus for the GtkWindow it's inside.
    widget must be a focusable widget, such as a GtkEntry; something like GtkFrame won't work.
    More precisely, it must have the GTK_CAN_FOCUS flag set.
    Use gtk_widget_set_can_focus() to modify that flag.
 */
FALCON_FUNC Widget::grab_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_grab_focus( (GtkWidget*)_obj );
}


/*#
    @method grab_default gtk.Widget
    @brief Causes widget to become the default widget.
    widget must have the GTK_CAN_DEFAULT flag set; typically you have to set this flag
    yourself by calling gtk_widget_set_can_default (widget, TRUE). The default widget
    is activated when the user presses Enter in a window. Default widgets must be
    activatable, that is, gtk_widget_activate() should affect them.
 */
FALCON_FUNC Widget::grab_default( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_grab_default( (GtkWidget*)_obj );
}


/*#
    @method set_name gtk.Widget
    @brief Attribute a name to the widget.
    @param name (string)
    Widgets can be named, which allows you to refer to them from a gtkrc file.
    You can apply a style to widgets with a particular name in the gtkrc file.
    See the documentation for gtkrc files (on the same page as the docs for GtkRcStyle).
    Note that widget names are separated by periods in paths (see gtk_widget_path()),
    so names with embedded periods may cause confusion.
 */
FALCON_FUNC Widget::set_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || i_name->isNil() || !i_name->isString() )
    {
        throw_inv_params( "S" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString s( i_name->asString() );
    gtk_widget_set_name( (GtkWidget*)_obj, s.c_str() );
}


/*#
    @method get_name gtk.Widget
    @brief Get the name of the widget.
    @return (string)
    See gtk.Widget.set_name() for the significance of widget names.
 */
FALCON_FUNC Widget::get_name( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* s = gtk_widget_get_name( (GtkWidget*)_obj );
    vm->retval( String( s ) );
}


/*#
    @method set_sensitive gtk.Widget
    @brief Sets the sensitivity of a widget.
    A widget is sensitive if the user can interact with it.
    Insensitive widgets are "grayed out" and the user can't interact with them.
    Insensitive widgets are known as "inactive", "disabled", or "ghosted" in some other toolkits.
 */
FALCON_FUNC Widget::set_sensitive( VMARG )
{
    Item* i_sens = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sens || i_sens->isNil() || !i_sens->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_set_sensitive( (GtkWidget*)_obj, i_sens->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_toplevel gtk.Widget
    @brief This function returns the topmost widget in the container hierarchy widget is a part of.
    @return (widget)
    If widget has no parent widgets, it will be returned as the topmost widget.
    No reference will be added to the returned widget; it should not be unreferenced.
 */
FALCON_FUNC Widget::get_toplevel( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* gwdt = gtk_widget_get_toplevel( (GtkWidget*)_obj );
    Item* wki = vm->findWKI( "Widget" );
    vm->retval( new Widget( wki->asClass(), gwdt ) );
}


/*#
    @method get_events gtk.Widget
    @brief Returns the event mask for the widget.
    @return (integer)
    (A bitfield containing flags from the GdkEventMask enumeration.)
    These are the events that the widget will receive.
 */
FALCON_FUNC Widget::get_events( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_widget_get_events( (GtkWidget*)_obj ) );
}


/*#
    @method is_ancestor gtk.Widget
    @brief Determines whether widget is somewhere inside ancestor, possibly with intermediate containers.
    @return (boolean)
 */
FALCON_FUNC Widget::is_ancestor( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() ||
        !( i_wdt->isOfClass( "Widget" ) || i_wdt->isOfClass( "gtk.Widget" ) ) )
    {
        throw_inv_params( "Widget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    vm->retval( (bool) gtk_widget_is_ancestor( (GtkWidget*)_obj, wdt ) );
}


/*#
    @method hide_on_delete gtk.Widget
    @brief Utility function.
    @return (boolean) always true
    Intended to be connected to the "delete-event" signal on a GtkWindow.
    The function calls gtk_widget_hide() on its argument, then returns TRUE.
    If connected to ::delete-event, the result is that clicking the close button
    for a window (on the window frame, top right corner usually) will hide but
    not destroy the window.
    By default, GTK+ destroys windows when ::delete-event is received.
 */
FALCON_FUNC Widget::hide_on_delete( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gboolean b = gtk_widget_hide_on_delete( (GtkWidget*)_obj );
    vm->retval( (bool) b );
}


} // Gtk
} // Falcon
