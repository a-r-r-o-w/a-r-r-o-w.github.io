(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[895],{44:function(e,s,n){Promise.resolve().then(n.bind(n,5314))},5314:function(e,s,n){"use strict";n.r(s),n.d(s,{default:function(){return i}});var r=n(7437),t=n(2265),a=n(6691),c=n.n(a),l=function(){arguments.length>0&&void 0!==arguments[0]&&arguments[0];let e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:-1;return(0,r.jsx)("img",{className:"missing-image-gradient cards__img",style:{"--gradient-angle":e%2==0?"45deg":"135deg"}})};function i(e){let{url:s,name:n,description:a,languages:i,image:d,video:u,index:o}=e,[m,h]=(0,t.useState)(!1),g=(0,t.useRef)(null);console.log("langs",i),(0,t.useEffect)(()=>{h(""!==document.createElement("video").canPlayType("video/mp4"))},[]);let p=l(n,o);return null!==d?p=(0,r.jsx)(c(),{className:"cards__img",src:d,alt:n,width:300,height:300}):null!==u&&(u.endsWith("gif")?p=(0,r.jsx)(c(),{className:"cards__img",src:u,alt:n,width:300,height:300}):m&&(p=(0,r.jsxs)("video",{className:"cards__vid",ref:g,loop:!0,muted:!0,playsInline:!0,alt:n,children:[u.endsWith("mp4")?(0,r.jsx)(r.Fragment,{children:(0,r.jsx)("source",{src:u,type:"video/mp4"})}):(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)("source",{src:u+".webm",type:"video/webm"}),(0,r.jsx)("source",{src:u+".mp4",type:"video/mp4"})]}),"Your browser does not support the video tag."]}))),(0,r.jsx)("div",{className:"cards__project",onMouseEnter:()=>{null!==g.current&&g.current.play()},onMouseLeave:()=>{null!=g.current&&(g.current.pause(),g.current.currentTime=0)},children:(0,r.jsxs)("a",{href:s,children:[(0,r.jsx)(t.Suspense,{fallback:l(o),children:p}),(0,r.jsxs)("div",{className:"cards__text",children:[(0,r.jsx)("div",{className:"font-bold text-xl;",children:n}),(0,r.jsx)("div",{className:"text-base",id:"description",children:a}),(0,r.jsx)("div",{className:"cards__languages",children:i.map((e,s)=>(0,r.jsx)("span",{className:"cards__language_container",children:e},s))})]})]})})}}},function(e){e.O(0,[986,971,938,744],function(){return e(e.s=44)}),_N_E=e.O()}]);